import numpy as np

from CAT.strategy.abstract_strategy import AbstractStrategy
from CAT.model import AbstractModel
from CAT.dataset import AdapTestDataset


class MAATStrategy(AbstractStrategy):
    def __init__(self, n_candidates=1):
        super().__init__()
        self.n_candidates = n_candidates

    @property
    def name(self):
        return 'Model Agnostic Adaptive Testing'

    def _compute_coverage_gain(self, sid, qid, adaptest_data: AdapTestDataset):
        concept_cnt = {}
        for q in adaptest_data.data[sid]:
            for c in adaptest_data.concept_map[q]:
                concept_cnt[c] = 0
        for q in list(adaptest_data.tested[sid]) + [qid]:
            for c in adaptest_data.concept_map[q]:
                concept_cnt[c] += 1
        return (sum(cnt / (cnt + 1) for c, cnt in concept_cnt.items()) /
                sum(1 for c in concept_cnt))

    def adaptest_select(self,
                        model: AbstractModel,
                        sid,
                        adaptest_data: AdapTestDataset,
                        item_candidates=None):
        assert hasattr(model, 'expected_model_change'), \
            'the models must implement expected_model_change method'
        pred_all = model.get_pred(adaptest_data)
        # model.model.get_knowledge_status(sid)
        # model.model.get_exer_params(adaptest_data.untested[sid][0])
        if item_candidates is None:
            untested_questions = np.array(list(adaptest_data.untested[sid]))
        else:
            available = adaptest_data.untested[sid].intersection(
                set(item_candidates))
            untested_questions = np.array(list(available))
        # untested_questions = np.array(list(adaptest_data.untested[sid]))
        fuzzy_order = sorted([(abs(pred_all[sid][qid]-0.5), qid) for qid in untested_questions])
        untested_questions =np.array( [ q for _, q in fuzzy_order[:100]])
        emc_arr = [
            model.expected_model_change1(sid, qid, adaptest_data, pred_all)
            for qid in untested_questions
        ]
        # emc_arr
        # candidates = untested_questions[np.argsort(emc_arr)[::1]
        #                                 [:self.n_candidates]]
        candidates = untested_questions[np.argsort(emc_arr)[::-1]
                                        [:self.n_candidates]]
        # print(candidates)
        # selection[sid] = max(candidates, key=lambda qid: self._compute_coverage_gain(sid, qid, adaptest_data))
        # print(emc_arr)
        return max(candidates,
                   key=lambda qid: self._compute_coverage_gain(
                       sid, qid, adaptest_data))
