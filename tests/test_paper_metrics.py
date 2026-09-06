import importlib.util
from pathlib import Path

spec=importlib.util.spec_from_file_location('metrics',Path(__file__).parents[1]/'scripts/moleg_paper_metrics.py')
metrics=importlib.util.module_from_spec(spec)
spec.loader.exec_module(metrics)


def test_empty_reference_is_unscorable_not_perfect():
    assert metrics.id_recall([],[]) is None
    assert metrics.id_recall(['a'],[])==0


def test_parent_and_chunk_relevance_are_separate():
    case={'source_id':'123_2','source_law_id':'123','reference_answer':'a grounded answer span'}
    score=metrics.known_item_metrics(['123_1','999_2','123_2'],case)
    assert score['chunk_hit1']==0 and score['law_hit1']==1
    assert score['chunk_mrr20']==1/3


def test_named_law_question_does_not_invent_unique_chunk_gold():
    score=metrics.known_item_metrics(['123_1'],{'source_id':'123_2','source_law_id':'123'})
    assert score['chunk_hit1'] is None and score['law_hit1']==1


def test_paired_bootstrap_identical_has_zero_delta():
    result=metrics.paired_cluster_ci([(1,1),(20,20)],resamples=100)
    assert result['delta_95ci']==[0,0]
    assert result['reduction_pct']==0


def test_ratio_of_means_not_mean_of_ratios():
    result=metrics.paired_cluster_ci([(1,.5),(9,9)],resamples=100)
    assert abs(result['speedup']-10/9.5)<1e-12


def test_mcnemar_uses_discordant_queries():
    result=metrics.mcnemar_exact([(0.,1.)]*10+[(1.,1.)]*100)
    assert result['p_two_sided']==2/1024
    assert result['wins']==10 and result['losses']==0


def test_percentile_interpolates_and_handles_empty():
    assert metrics.percentile([], .95) is None
    assert metrics.percentile([0,10], .95)==9.5


def test_holm_correction_is_monotonic():
    adjusted=metrics.holm_adjust({'a':.02,'b':.012,'c':.012})
    assert all(abs(v-.036)<1e-12 for v in adjusted.values())


def test_rare_tail_is_visible_even_below_one_percent():
    result=metrics.timing([1]*399+[250])
    assert result['p99_s']==1
    assert result['max_s']==250 and result['over_120_s_count']==1
