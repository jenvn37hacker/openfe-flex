"""Independent schedule checks and numerical comparisons with pinned upstream."""
from copy import deepcopy
from importlib.util import find_spec
import json
import random
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from openfe_flex import OpenFE, LightGBMEvaluator, numeric_candidates, transform_manifest
from openfe_flex._upstream.FeatureGenerator import Node, FNode
from openfe_flex._upstream.openfe import OpenFE as UpstreamOpenFE
from openfe_flex._upstream.utils import tree_to_formula
from test_injection import CustomEvaluator
from test_catboost_backend import dataset


class ZeroEvaluator(CustomEvaluator):
    def predict_raw(self, model, data, *, baseline=None):
        return np.zeros((len(data),model["dimensions"])) if baseline is None else np.asarray(baseline)


class ScriptedSearch(OpenFE):
    def _calculate_and_evaluate(self, candidates, train, val):
        self.trace.append((len(train),len(val),[tree_to_formula(n) for n in candidates]))
        return [(node,self.scores[tree_to_formula(node)]+len(train)*1e-5) for node in candidates]


class SerialReference(UpstreamOpenFE):
    """Only removes multiprocessing; original scoring/pruning/attribution remain."""
    def _calculate(self,candidates,train,val):
        for node in candidates:
            node.calculate(self.data.loc[list(train)+list(val)],is_root=True)
            node.f_delete()
        return candidates

    def _calculate_and_evaluate(self,candidates,train,val):
        self.trace.append((len(train),len(val),[tree_to_formula(n) for n in candidates]))
        train_y,val_y=self.label.loc[train],self.label.loc[val]
        train_init,val_init=self.init_scores.loc[train],self.init_scores.loc[val]
        initial=self.get_init_metric(val_init,val_y)
        output=[]
        for node in candidates:
            node.calculate(self.data.loc[list(train)+list(val)],is_root=True)
            score=self._evaluate(node,train_y,val_y,train_init,val_init,initial)
            node.delete()
            output.append((node,score))
        self.evaluations.append({tree_to_formula(node):score for node,score in output})
        return output


class PaperCoreTests(unittest.TestCase):
    def test_full_numeric_enumeration_and_optional_budget(self):
        formulas={tree_to_formula(n) for n in numeric_candidates(["a","b"])}
        self.assertEqual(len(formulas),22)  # 7*2 unary + 4 symmetric + 2*2 directional
        self.assertTrue({"(a/b)","(b/a)","(a-b)","(b-a)","sigmoid(a)","min(a,b)"}<=formulas)
        self.assertEqual(len(numeric_candidates(["a","b"],max_candidates=3)),3)
        for invalid in (0,-1,True,1.5):
            with self.assertRaises(ValueError):
                numeric_candidates(["a","b"],max_candidates=invalid)

    def test_halving_schedule_matches_upstream_on_positive_candidates(self):
        nodes=numeric_candidates(["a","b"],operators=("abs","square","+","*","/"))
        scores={tree_to_formula(n):float(i+1) for i,n in enumerate(nodes)}
        ours=ScriptedSearch(evaluator=CustomEvaluator())
        ours.trace=[];ours.scores=scores
        ours.train_index=pd.Index(range(64));ours.val_index=pd.Index(range(64,96))
        ours.n_data_blocks=4;ours.min_candidate_features=1;ours.candidate_features_list=deepcopy(nodes)
        ours.verbose=False;ours.task="regression"
        reference=UpstreamOpenFE()
        for name in ("train_index","val_index","n_data_blocks","min_candidate_features","verbose"):
            setattr(reference,name,getattr(ours,name))
        reference.is_stage1=True;reference.candidate_features_list=deepcopy(nodes)
        reference._calculate_and_evaluate=ours._calculate_and_evaluate
        random.seed(42)
        expected=reference.stage1_select();trace=ours.trace.copy();ours.trace=[]
        random.seed(42)
        actual=ours.stage1_select()
        self.assertEqual(ours.trace,trace)
        self.assertEqual([tree_to_formula(n) for n in actual],[tree_to_formula(n) for n in expected])
        self.assertEqual([r[0] for r in trace],[16,32,64])

    def test_nonpositive_candidates_return_empty_and_transform_safely(self):
        x,y,train,val=dataset(3)
        evaluator=ZeroEvaluator()
        generator=OpenFE(evaluator=evaluator)
        result=generator.fit(x,y,train_index=train,val_index=val,n_data_blocks=1,
                             operators=("abs",),verbose=False)
        self.assertEqual(result,[])
        self.assertIsNone(generator.stage2_model_)
        self.assertNotIn("stage2",evaluator.stages)
        self.assertEqual(generator.transform_new(x).shape,(len(x),0))
        augmented,_=generator.transform(x,x,[])
        pd.testing.assert_frame_equal(augmented,x)
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/"run.json"
            generator.save_run(path)
            metadata=json.loads(path.read_text())
            self.assertEqual(metadata["orders"][0]["positive_survivors"],0)

    def test_higher_orders_extend_basis_and_flatten_dependencies(self):
        x,y,train,val=dataset(3)
        generator=OpenFE(evaluator=CustomEvaluator())
        ranked=generator.fit(x,y,train_index=train,val_index=val,max_order=2,
                             n_new_features=1,max_candidates=None,operators=("abs","+"),
                             n_data_blocks=1,verbose=False)
        self.assertEqual(len(generator.order_history_),2)
        self.assertEqual(len(ranked),2)
        manifest=generator.get_manifest()
        self.assertTrue(all(set(entry["source_features"])<=set(x.columns) for entry in manifest))
        self.assertFalse(any("__openfe_generated" in entry["formula"] for entry in manifest))
        self.assertGreater(generator.order_history_[1]["candidate_count"],0)
        transformed=transform_manifest(x,json.loads(json.dumps(manifest)))
        pd.testing.assert_frame_equal(transformed,generator.transform_new(x))
        single=pd.concat([transform_manifest(x.iloc[[i]],manifest) for i in range(len(x))])
        pd.testing.assert_frame_equal(single,transformed)
        self.assertEqual(len(generator.stage2_columns_),
                         generator.order_history_[1]["positive_survivors"]+x.shape[1]+1)

    def test_hierarchy_stops_when_no_candidate_improves(self):
        x,y,train,val=dataset(3)
        generator=OpenFE(evaluator=ZeroEvaluator())
        self.assertEqual(generator.fit(x,y,train_index=train,val_index=val,max_order=3,
                                       max_candidates=3,n_data_blocks=1,verbose=False),[])
        self.assertEqual(len(generator.order_history_),1)

    def test_tiny_classification_blocks_enlarge_before_initial_deduplication(self):
        x,y,train,val=dataset(31)
        generator=OpenFE(evaluator=CustomEvaluator())
        generator.fit(x,y,train_index=train,val_index=val,n_data_blocks=16,
                      operators=("abs",),verbose=False)
        self.assertGreaterEqual(generator.pruning_history_[0]["train_rows"],31)
        # Every oracle correction has equal improvement; initial deduplication
        # must still run even when the earliest data blocks were skipped.
        self.assertEqual(len(generator.pruning_history_[0]["scores"]),1)

    def test_invalid_blocks_and_higher_order_settings(self):
        x,y,train,val=dataset(3)
        for kwargs in ({"n_data_blocks":3},{"max_order":0},{"max_order":2,"n_new_features":None},
                       {"max_order":2,"init_scores":np.zeros((len(x),3))}):
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    OpenFE(evaluator=CustomEvaluator()).fit(x,y,train_index=train,val_index=val,**kwargs)

    def test_paper_lightgbm_configuration(self):
        evaluator=LightGBMEvaluator.paper()
        self.assertEqual(evaluator.params["learning_rate"],.1)
        self.assertEqual(evaluator.params["num_leaves"],16)
        self.assertEqual(evaluator.stage1_params["n_estimators"],1000)

    @unittest.skipUnless(find_spec("lightgbm") is not None,"LightGBM not installed")
    def test_native_lightgbm_scores_pruning_and_gain_match_upstream(self):
        rng=np.random.default_rng(17)
        # Each initial block has enough rows to fit splits under min_child=20;
        # the squared component supplies a clearly positive candidate so this
        # parity check does not exercise upstream's deliberate fallback mismatch.
        x=pd.DataFrame(rng.normal(size=(384,2)),columns=["a","b"])
        y=pd.DataFrame(x.a**2+.25*x.a*x.b+.1*rng.normal(size=len(x)))
        nodes=numeric_candidates(x.columns,operators=("abs","square","+","*","/"))
        common=dict(n_estimators=100,learning_rate=.1,num_leaves=16,min_child_samples=20,
                    seed=1,n_jobs=1,verbosity=-1,deterministic=True,force_col_wise=True)
        stage2=dict(common,n_estimators=20,importance_type="gain",metric="rmse")
        settings=dict(task="regression",train_index=x.index[:288],val_index=x.index[288:],
                      init_scores=pd.DataFrame(np.zeros(len(x)),index=x.index),
                      feature_boosting=True,stage2_metric="gain_importance",n_data_blocks=4,
                      min_candidate_features=1,n_jobs=1,seed=1,verbose=False)
        reference=SerialReference();reference.trace=[];reference.evaluations=[]
        with tempfile.TemporaryDirectory() as folder:
            reference.fit(x.copy(),y.copy(),candidate_features_list=deepcopy(nodes),
                          stage2_params=stage2,tmp_save_path=str(Path(folder)/"data.feather"),**settings)
        generator=OpenFE(evaluator=LightGBMEvaluator(params=common,stage1_params={"n_estimators":100},stage2_params=stage2))
        generator.fit(x.copy(),y.copy(),candidate_features_list=deepcopy(nodes),n_new_features=None,**settings)
        self.assertEqual([(r["train_rows"],r["validation_rows"]) for r in generator.pruning_history_],
                         [(r[0],r[1]) for r in reference.trace])
        for actual,expected in zip(generator.pruning_history_,reference.evaluations):
            for entry in actual["scores"]:
                # Native validation evaluation and independently recomputed loss
                # differ slightly because LightGBM internally casts values.
                self.assertAlmostEqual(entry["improvement"],expected[entry["formula"]],delta=1e-6)
        actual={tree_to_formula(n):score for n,score in generator.new_features_scores_list}
        expected={tree_to_formula(n):score for n,score in reference.new_features_scores_list}
        self.assertEqual(actual.keys(),expected.keys())
        for formula in actual:
            self.assertAlmostEqual(actual[formula],expected[formula],places=7)

