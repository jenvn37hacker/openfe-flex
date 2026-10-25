import json
from importlib.util import find_spec
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

import numpy as np
import pandas as pd
from openfe_flex import CatBoostOpenFE, LightGBMOpenFE, numeric_candidates, transform_manifest
from openfe_flex._upstream.FeatureGenerator import Node, FNode
from openfe_flex._upstream.utils import tree_to_formula


def dataset(classes):
    rng = np.random.default_rng(42)
    y = np.tile(np.arange(classes), 8)
    x = pd.DataFrame(rng.normal(size=(len(y), 4)),
                     columns=["S1_rms", "S1_std", "S2_rms", "S2_std"],
                     index=np.arange(len(y))*3+100)
    x.iloc[:, 0] += y
    mask = np.arange(len(y)) < classes*6
    return x, pd.Series(y, index=x.index), x.index[mask], x.index[~mask]


@unittest.skipUnless(find_spec("catboost") is not None, "CatBoost backend is not installed")
class CatBoostBackendTests(unittest.TestCase):
    def fit_case(self, classes):
        x, y, train, val = dataset(classes)
        model = CatBoostOpenFE()
        features = model.fit(
            x, y, train_index=train, val_index=val,
            candidate_features_list=numeric_candidates(x.columns, max_candidates=6),
            model_params={"iterations":10,"depth":2},
            stage1_params={"iterations":8}, n_data_blocks=1,
            min_candidate_features=2, n_jobs=1, verbose=False)
        from catboost import CatBoostClassifier
        self.assertIsInstance(model.stage2_model_, CatBoostClassifier)
        self.assertGreater(len(features),0)
        self.assertEqual(model.init_scores.shape,(len(x),classes if classes>2 else 1))
        self.assertTrue(np.isfinite(model.init_scores).all().all())
        return model, features, x

    def test_binary_17_and_31_classes_and_serialization(self):
        for classes in (2,17,31):
            with self.subTest(classes=classes):
                model, features, x = self.fit_case(classes)
                chosen = features[:3]
                transformed = model.transform_new(x,chosen)
                self.assertEqual(transformed.shape,(len(x),len(chosen)))
                self.assertFalse(set(transformed.columns)&set(x.columns))
                manifest = model.get_manifest(chosen)
                restored = transform_manifest(x,json.loads(json.dumps(manifest)))
                pd.testing.assert_frame_equal(transformed,restored)
                train_aug,test_aug=model.transform(x.iloc[:4],x.iloc[4:7],chosen,n_jobs=1)
                self.assertEqual(train_aug.shape,(4,x.shape[1]+len(chosen)))
                np.testing.assert_allclose(test_aug.iloc[:,x.shape[1]:],transformed.iloc[4:7],equal_nan=True)
                singles = pd.concat([transform_manifest(x.iloc[[i]],manifest) for i in range(len(x))])
                pd.testing.assert_frame_equal(transformed,singles)
                # SHAP aggregation must account for class and sample axes, excluding bias.
                self.assertEqual(len(model.stage2_importances_),len(model.stage2_columns_))
                self.assertTrue(np.isfinite(model.stage2_importances_).all())
                self.assertTrue((model.stage2_importances_>=0).all())
                with tempfile.TemporaryDirectory() as folder:
                    path=Path(folder)/"formulas.json"
                    model.save_manifest(path,chosen)
                    self.assertEqual(json.loads(path.read_text()),manifest)

    def test_correction_score_matches_native_validation_loss(self):
        model,features,x=self.fit_case(3)
        captured=[]
        factory=model.evaluator.fit
        def capture(*args,**kwargs):
            result=factory(*args,**kwargs); captured.append(result); return result
        model.evaluator.fit=capture
        feature=features[0]
        feature.calculate(x,is_root=True)
        train_y=model.label.loc[model.train_index]
        val_y=model.label.loc[model.val_index]
        train_init=model.init_scores.loc[model.train_index]
        val_init=model.init_scores.loc[model.val_index]
        before=model.get_init_metric(val_init,val_y)
        score=model._evaluate(feature,train_y,val_y,train_init,val_init,before)
        native_loss=captured[-1].get_best_score()["validation"]["MultiClass"]
        self.assertAlmostEqual(score,before-native_loss,places=6)
        values=pd.DataFrame(feature.data.loc[model.val_index])
        raw=captured[-1].predict(model.evaluator.pool(values),prediction_type="RawFormulaVal")
        with_baseline=captured[-1].predict(model.evaluator.pool(values,baseline=val_init),prediction_type="RawFormulaVal")
        np.testing.assert_allclose(with_baseline,raw+val_init.to_numpy(),atol=2e-7,rtol=0)

    def test_candidate_budget_and_direction(self):
        first=numeric_candidates(["a","b"],max_candidates=20,operators=("/","-"))
        self.assertEqual({tree_to_formula(n) for n in first},{"(a/b)","(b/a)","(a-b)","(b-a)"})
        a=numeric_candidates([f"S{i}_rms" for i in range(50)],max_candidates=37)
        b=numeric_candidates([f"S{i}_rms" for i in range(50)],max_candidates=37)
        self.assertEqual(len(a),37)
        self.assertEqual([tree_to_formula(n) for n in a],[tree_to_formula(n) for n in b])

    def test_heldout_data_cannot_change_transforms(self):
        x,_,_,_=dataset(3)
        manifest=[{"name":"autoFE_0000","formula":"(S1_rms/S2_std)",
                   "source_features":["S1_rms","S2_std"]}]
        before=transform_manifest(x,manifest)
        extreme=x.iloc[:1].copy(); extreme.iloc[:,:]=1e8; extreme.index=[-1]
        together=transform_manifest(pd.concat([x,extreme]),manifest)
        pd.testing.assert_frame_equal(before,together.loc[x.index])
        with self.assertRaises(ValueError):
            transform_manifest(x,[{"name":"bad","formula":"freq(S1_rms)","source_features":["S1_rms"]}])

    def test_schema_and_split_validation(self):
        x,y,train,val=dataset(3)
        bad=Node("GroupByThenMean",[FNode("S1_rms"),FNode("S1_std")])
        with self.assertRaises(ValueError):
            CatBoostOpenFE().fit(x,y,candidate_features_list=[bad])
        with self.assertRaises(ValueError):
            CatBoostOpenFE().fit(x,y,train_index=train,val_index=train,feature_boosting=False)
        with self.assertRaises(ValueError):
            CatBoostOpenFE().fit(x,y.iloc[::-1])

    @unittest.skipUnless(find_spec("lightgbm") is not None, "optional LightGBM backend is not installed")
    def test_lightgbm_backend_still_works(self):
        x,y,train,val=dataset(3)
        model=LightGBMOpenFE()
        result=model.fit(x.copy(),y.to_frame().copy(),task="classification",
                         train_index=train,val_index=val,feature_boosting=False,
                         candidate_features_list=numeric_candidates(x.columns,max_candidates=3),
                         n_data_blocks=1,min_candidate_features=2,n_jobs=1,verbose=False,
                         stage2_params={"n_estimators":8,"num_leaves":3,"n_jobs":1,"verbosity":-1})
        self.assertGreater(len(result),0)
        self.assertEqual(model.stage2_metric,"gain_importance")

    def test_nonfinite_constant_candidate_is_skipped(self):
        x,y,train,val=dataset(3)
        model=CatBoostOpenFE()
        feature=Node("square",[FNode("S1_rms")])
        feature.data=pd.Series(np.resize([np.inf,-np.inf],len(x)),index=x.index)
        self.assertEqual(model._evaluate(feature,y.loc[train],y.loc[val],None,None,1.0),0.0)

    def test_missing_optional_backend_has_install_hint(self):
        with patch("openfe_flex.lightgbm_backend.find_spec",return_value=None):
            with self.assertRaisesRegex(ImportError,r"openfe-flex\[lightgbm\]"):
                LightGBMOpenFE().fit(None,None)

    def test_parallel_and_sequential_scores_agree(self):
        x,y,train,val=dataset(3)
        scores=[]
        for jobs in (1,2):
            model=CatBoostOpenFE()
            model.fit(x,y,train_index=train,val_index=val,feature_boosting=False,
                      candidate_features_list=numeric_candidates(x.columns,max_candidates=4),
                      model_params={"iterations":8,"depth":2},stage1_params={"iterations":8},
                      n_data_blocks=1,min_candidate_features=2,n_jobs=jobs,verbose=False)
            scores.append({tree_to_formula(n):score for n,score in model.new_features_scores_list})
        self.assertEqual(set(scores[0]),set(scores[1]))
        for key in scores[0]:
            self.assertAlmostEqual(scores[0][key],scores[1][key],places=10)

    def test_regression_and_invalid_oof_folds(self):
        x,y,train,val=dataset(3)
        model=CatBoostOpenFE()
        result=model.fit(x,y.astype(float),task="regression",train_index=train,val_index=val,
                         candidate_features_list=numeric_candidates(x.columns,max_candidates=4),
                         model_params={"iterations":8,"depth":2},stage1_params={"iterations":8},
                         n_data_blocks=1,min_candidate_features=2,n_jobs=1,verbose=False)
        self.assertGreater(len(result),0)
        self.assertEqual(model.init_scores.shape,(len(x),1))
        with self.assertRaisesRegex(ValueError,"OOF"):
            CatBoostOpenFE().fit(x,y,train_index=train,val_index=val,
                                baseline_cv=[(np.arange(10),np.arange(10))],
                                model_params={"iterations":1},max_candidates=2,verbose=False)

    def test_multiclass_constant_baseline_is_log_prior(self):
        model,_,x=self.fit_case(3)
        model.feature_boosting=False
        baseline=model.get_init_score(None)
        np.testing.assert_allclose(baseline,-np.log(3))
        with self.assertRaises(ValueError):
            CatBoostOpenFE().fit(x,np.tile(np.arange(3),8),metric="rmse")


if __name__=="__main__":
    unittest.main()
