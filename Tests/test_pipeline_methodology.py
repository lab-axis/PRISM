"""Method/flow contracts independent of historical experiment results.

These tests construct inputs and models but do not fit a training epoch.
Full checkpoint replay remains in the stage-specific integration tests.
"""
import ast
from dataclasses import replace
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest
import torch

ROOT=Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def modules():
    loaded={}
    for stage in range(4,10):
        path=next((ROOT/"Sources").glob(f"{stage:02d}_*.py"))
        spec=importlib.util.spec_from_file_location(f"method_contract_{stage}",path)
        module=importlib.util.module_from_spec(spec)
        sys.modules[spec.name]=module
        spec.loader.exec_module(module)
        loaded[stage]=module
    return loaded


@pytest.fixture
def inputs(modules,monkeypatch):
    m4,m5=modules[4],modules[5]
    rng=np.random.default_rng(56)
    names=["history_"+a for a in ("profitability","productivity","growth")]
    for suffix in ("lag02","lag04","change04"):
        names.extend(f"targethist_{a}_{suffix}" for a in ("profitability","productivity","growth"))
    names.extend(f"feature{i}" for i in range(len(names),157))
    panel=m4.Panel(rng.normal(size=(68,59,157)),rng.normal(size=(68,59,3)),
                   pd.period_range("2006Q1","2022Q4",freq="Q").astype(str).tolist(),
                   list(map(str,range(59))),list(map(str,range(59))),names,56,{})
    prior=rng.normal(size=(68,59,3))
    monkeypatch.setattr(m5,"stage04",lambda:m4)
    monkeypatch.setattr(m4,"make_fold",lambda p:(prior,None))
    return panel,prior,m5.Inputs(panel)


def test_no_tuned_values_or_directory_changes(modules):
    m=modules[4]
    assert (m.STAGE04_ACTIVE_WIDTH,m.FIXED_LEARNING_RATE,m.STAGE04_ACTIVE_EPOCHS)==(16,.001,75)
    assert m.STAGE04_FIXED_ALPHA==(.25,.1,.25)
    assert m.STAGE04_ROBUST_SNAPSHOTS==(50,75,100)
    for stage,module in modules.items():
        assert module.RESULTS==ROOT/"Results"/Path(module.__file__).stem
    code=ast.parse(Path(m.__file__).read_text(encoding="utf8"))
    freeze=next(n for n in code.body if isinstance(n,ast.FunctionDef) and n.name=="_freeze_dev")
    assignment=next(n for n in ast.walk(freeze) if isinstance(n,ast.Assign)
                    and any(isinstance(t,ast.Name) and t.id=="mode" for t in n.targets))
    assert isinstance(assignment.value,ast.Call)
    assert assignment.value.func.id=="select_bmtgnn_mode"


@pytest.mark.parametrize("length",(1,5,12,20))
def test_common_search_then_full_support_refit(modules,inputs,length):
    _,_,maker=inputs
    common=maker.fold(length)
    full=maker.fold(length,full_train=True)
    assert common["train_origins"].tolist()==list(range(20,56))
    assert full["train_origins"].tolist()==list(range(max(length,5),56))
    for key in ("train_x","train_mask","train_y","train_last","train_structure","train_context"):
        np.testing.assert_array_equal(full[key][-36:],common[key])
    for key in ("val_x","val_mask","val_y","val_last","val_structure","val_context"):
        np.testing.assert_array_equal(full[key],common[key])


@pytest.mark.parametrize("mode",("direct","residual","structural"))
def test_initial_prediction_is_calibrated_anchor_not_random(modules,inputs,mode):
    _,_,maker=inputs
    data=maker.fold(1,full_train=True)
    net=modules[5].instantiate(modules[4],"B-MTGNN",2026,1,data,mode).eval()
    assert sum(p.numel() for p in net.parameters())==67190
    assert net.anchor_bias.shape==(59,3) and "anchor_bias" not in dict(net.named_parameters())
    z={name:torch.as_tensor(data["val_"+name][:1]) for name in ("x","mask","last","structure","context")}
    with torch.inference_mode():
        prediction=net(z["x"],z["mask"],torch.eye(59),z["last"],z["structure"],z["context"])
    anchor=(torch.zeros_like(z["last"]) if mode=="direct" else z["last"] if mode=="residual" else z["structure"])
    torch.testing.assert_close(prediction[:,0],anchor+net.anchor_bias,rtol=0,atol=1e-7)


def test_median_uses_only_train_and_is_l1_optimal(modules,inputs):
    panel,prior,_=inputs
    m4=modules[4]
    origins=m4.admissible_train_origins(1)
    biases=m4.train_anchor_biases(panel,prior,origins,np.zeros(3),np.ones(3))
    changed=replace(panel,y=panel.y.copy())
    changed.y[56:]+=1e7
    np.testing.assert_array_equal(biases,m4.train_anchor_biases(changed,prior,origins,np.zeros(3),np.ones(3)))
    residual=panel.y[origins]-prior[origins]
    best=np.abs(residual-biases[2]).sum(0)
    for shift in (-1.,1.):
        assert np.all(np.abs(residual-(biases[2]+shift)).sum(0)>=best-1e-6)
    with pytest.raises(ValueError):
        m4.train_anchor_biases(panel,prior,np.arange(5,57),np.zeros(3),np.ones(3))


def test_joint_loss_is_in_original_units(modules):
    prediction=torch.tensor([[[[1.,-2.,3.]]]])
    target=torch.zeros_like(prediction)
    scale=torch.tensor([2.,3.,4.])
    error=(prediction-target)*scale
    expected=(error.abs().mean()+error.mean(-1).abs().mean())/2
    torch.testing.assert_close(modules[4].joint_axis_ici_loss(prediction,target,scale),expected)


@pytest.mark.parametrize("length",(1,12,20))
def test_stage06_reference_is_identical_to_stage05_refit(modules,inputs,monkeypatch,length):
    panel,prior,maker=inputs
    m6=modules[6]
    monkeypatch.setattr(m6,"stage04",lambda:modules[4])
    monkeypatch.setattr(m6,"stage05",lambda:modules[5])
    monkeypatch.setattr(m6,"selected_lookback",lambda:length)
    assert m6.original_stage05_parity(panel,prior)
    actual,_=m6.prepare_policy(panel,m6.ORIGINAL,prior)
    expected=maker.fold(length,full_train=True)
    np.testing.assert_array_equal(actual["anchor_biases"],expected["anchor_biases"])


def test_pruning_preserves_original_indices_and_never_restores_removed_inputs(modules):
    m8=modules[8]
    parent=(2,10,91,130)
    ctx=dict(m07=modules[7],active_indices=parent)
    assert m8.topk_indices((91,2,130,10),2)==(2,91)
    for seed in m8.RANDOM_SUBSET_SEEDS:
        assert set(m8.random_indices(seed,parent)).issubset(parent)
    with pytest.raises(ValueError):
        m8.subset(ctx,(0,2),"invalid_reintroduced_feature")
    rows=[dict(K=k,axis_mae=2.,qualified=False) for k in range(1,5)]
    assert m8.select_k(rows,4)==4
    rows[1].update(axis_mae=1.,qualified=True)
    assert m8.select_k(rows,4)==2
    assert m8.select_k([dict(K=1,axis_mae=1.,qualified=False)],1)==1


@pytest.mark.parametrize("mode",("direct","residual","structural"))
def test_stage09_parent_initialization_survives_compact_input(modules,inputs,mode):
    _,_,maker=inputs
    m4,m5,m7,m9=(modules[i] for i in (4,5,7,9))
    data=maker.fold(1,full_train=True)
    indices=(2,10,91,130)
    subset=m7.Subset("parent",None,indices,tuple(i for i in range(157) if i not in indices),pd.DataFrame(),0)
    ctx=dict(m04=m4,m05=m5,m07=m7,lookback=1,output_mode=mode,
             full_fold=data,baseline_parameters=67190,selected_indices=indices,selected_subset=subset)
    selected=m7.adapt_inputs(data,subset,"compact")
    reference=m7.instantiate_subset(ctx,subset,2026,compact=True)
    reconstructed=m9.model_factory(ctx,16,2026,selected)
    for key,value in reference.state_dict().items():
        torch.testing.assert_close(value,reconstructed.state_dict()[key],rtol=0,atol=0)
    assert reconstructed.prediction_mode==mode


def test_no_alpha_search_when_selected_model_does_not_use_alpha(modules):
    m9=modules[9]
    assert len(m9.grid("structural"))==2268
    assert len(m9.grid("direct"))==len(m9.grid("residual"))==84
    assert m9.active_alphas("residual")== (m9.REFERENCE[2],)


def test_restart_cannot_target_data_or_other_projects(modules):
    for path in (ROOT/"Data",ROOT/"Results",ROOT.parent):
        with pytest.raises(ValueError):
            modules[4].reset_stage_results(path)
