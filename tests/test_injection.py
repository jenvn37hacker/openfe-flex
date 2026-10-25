"""Contract and numerical integration tests for the injected evaluator API."""
from importlib.util import find_spec
import json
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from openfe_flex import OpenFE, CatBoostEvaluator, LightGBMEvaluator, numeric_candidates, transform_manifest
from test_catboost_backend import dataset


class CustomEvaluator:
    """Deliberately unrelated to built-in evaluators and model libraries."""
    supported_importances = ("shap",)

    def __init__(self):
        self.stages = []

    def check_available(self):
        pass

    def fit(self, data, labels, *, validation, baseline, context, stage, early_stopping_rounds):
        self.stages.append(stage)
        # Oracle margins are a deterministic contract-test double, not an ML
        # estimator. They ensure positive corrections reach stage 2 without the
        # old nonpositive fallback or any optional model library.
        return {"columns": data.shape[1], "dimensions": context.n_classes if context.n_classes > 2 else 1,
                "validation_labels":dict(zip(validation[0].index,np.asarray(validation[1]).ravel()))}

    def predict_raw(self, model, data, *, baseline=None):
        labels=np.array([model["validation_labels"][index] for index in data.index],dtype=int)
        if model["dimensions"]==1:
            raw=(labels*2-1).reshape(-1,1)*2.0
        else:
            raw=np.eye(model["dimensions"])[labels]*2.0
        return raw if baseline is None else raw+np.asarray(baseline).reshape(raw.shape)

    def importance(self, model, data, *, method, context):
        return np.arange(data.shape[1],dtype=float)


class InjectionTests(unittest.TestCase):
    def fit_case(self, evaluator, classes=3, jobs=1, task="classification", metric=None, allow_empty=False):
        x,y,train,val=dataset(classes)
        generator=OpenFE(evaluator=evaluator)
        ranked=generator.fit(
            x,y,task=task,metric=metric,train_index=train,val_index=val,
            max_candidates=4,n_data_blocks=1,min_candidate_features=2,
            n_jobs=jobs,verbose=False)
        self.assertIs(generator.evaluator,evaluator)
        if not allow_empty:
            self.assertGreater(len(ranked),0)
        elif not ranked:
            self.assertIsNone(generator.stage2_model_)
        self.assertTrue(np.isfinite(generator.init_scores).all().all())
        generated=generator.transform_new(x,ranked[:2])
        pd.testing.assert_frame_equal(generated,transform_manifest(x,generator.get_manifest(ranked[:2])))
        self.assertEqual(generator.stage2_importances_.shape,(len(generator.stage2_columns_),))
        return generator,ranked,x

    def test_custom_evaluator_controls_every_stage(self):
        evaluator=CustomEvaluator()
        generator,_,_=self.fit_case(evaluator)
        self.assertEqual(set(evaluator.stages),{"baseline","stage1","stage2"})
        self.assertIsInstance(generator.stage2_model_,dict)
        with self.assertRaisesRegex(TypeError,"Evaluator protocol"):
            OpenFE(evaluator=object())
        with self.assertRaises(TypeError):
            OpenFE()

    def test_parallel_search_accepts_custom_evaluator(self):
        self.fit_case(CustomEvaluator(),jobs=2)

    def test_package_import_and_custom_evaluator_need_no_model_backend(self):
        # A fresh interpreter blocks either optional model library, even if installed.
        code='''import importlib.abc,sys
class BlockModels(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname.split('.')[0] in {'catboost','lightgbm'}:
            raise ImportError('model backend must not be imported')
sys.meta_path.insert(0,BlockModels())
from openfe_flex import OpenFE,CatBoostEvaluator,LightGBMEvaluator,transform_manifest
assert 'catboost' not in sys.modules and 'lightgbm' not in sys.modules
'''
        subprocess.run([sys.executable,"-I","-c",code],check=True,cwd="/tmp" if sys.platform!="win32" else None)

    def test_missing_backend_install_hints(self):
        for name,evaluator,extra in (("catboost",CatBoostEvaluator(),"catboost"),
                                     ("lightgbm",LightGBMEvaluator(),"lightgbm")):
            with patch(f"openfe_flex.evaluators.{name}.find_spec",return_value=None):
                with self.assertRaisesRegex(ImportError,rf"openfe-flex\[{extra}\]"):
                    self.fit_case(evaluator)

    def test_evaluator_method_and_parameter_validation(self):
        with self.assertRaisesRegex(ValueError,"stage2_metric"):
            x,y,_,_=dataset(3)
            OpenFE(evaluator=CustomEvaluator()).fit(x,y,stage2_metric="gain_importance")
        with self.assertRaisesRegex(TypeError,"on the evaluator"):
            OpenFE(evaluator=CustomEvaluator()).fit(x,y,model_params={})

    @unittest.skipUnless(find_spec("catboost") is not None,"CatBoost not installed")
    def test_catboost_injected_and_compatibility_paths_agree(self):
        from openfe_flex import CatBoostOpenFE
        from openfe_flex._upstream.utils import tree_to_formula
        params={"iterations":8,"depth":2}
        evaluator=CatBoostEvaluator(params=params,stage1_params={"iterations":8})
        generator,ranked,x=self.fit_case(evaluator)
        _,y,train,val=dataset(3)
        old=CatBoostOpenFE()
        old_ranked=old.fit(x,y,train_index=train,val_index=val,max_candidates=4,
                           n_data_blocks=1,min_candidate_features=2,n_jobs=1,verbose=False,
                           model_params=params,stage1_params={"iterations":8})
        self.assertEqual([tree_to_formula(n) for n in ranked],[tree_to_formula(n) for n in old_ranked])
        np.testing.assert_allclose(generator.stage2_importances_,old.stage2_importances_,atol=1e-12)
        self.assertEqual(params,{"iterations":8,"depth":2})

    @unittest.skipUnless(find_spec("lightgbm") is not None,"LightGBM not installed")
    def test_lightgbm_classification_regression_and_parallel(self):
        evaluator=LightGBMEvaluator(params={"n_estimators":8,"num_leaves":3},
                                    stage1_params={"n_estimators":8})
        for classes in (2,17,31):
            with self.subTest(classes=classes):
                model,_,_=self.fit_case(evaluator,classes=classes)
                self.assertEqual(model.init_scores.shape,(classes*8,classes if classes>2 else 1))
        self.fit_case(evaluator,task="regression")
        self.fit_case(evaluator,classes=2,metric="auc",allow_empty=True)
        self.fit_case(evaluator,jobs=2)

    @unittest.skipUnless(find_spec("lightgbm") is not None,"LightGBM not installed")
    def test_lightgbm_raw_baseline_is_added_once_and_loss_matches_native(self):
        evaluator=LightGBMEvaluator(params={"n_estimators":10,"num_leaves":3},
                                    stage1_params={"n_estimators":8})
        generator,features,x=self.fit_case(evaluator)
        captured=[]
        original=evaluator.fit
        def capture(*args,**kwargs):
            model=original(*args,**kwargs);captured.append(model);return model
        evaluator.fit=capture
        feature=features[0]
        feature.calculate(x,is_root=True)
        train_y=generator.label.loc[generator.train_index]
        val_y=generator.label.loc[generator.val_index]
        train_init=generator.init_scores.loc[generator.train_index]
        val_init=generator.init_scores.loc[generator.val_index]
        before=generator.get_init_metric(val_init,val_y)
        score=generator._evaluate(feature,train_y,val_y,train_init,val_init,before)
        model=captured[-1]
        self.assertAlmostEqual(score,before-model.best_score_["valid_0"]["multi_logloss"],places=6)
        values=pd.DataFrame(feature.data.loc[generator.val_index])
        raw=evaluator.predict_raw(model,values)
        corrected=evaluator.predict_raw(model,values,baseline=val_init)
        np.testing.assert_allclose(corrected,raw+val_init.to_numpy(),atol=1e-12)

