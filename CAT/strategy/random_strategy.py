from matplotlib.style import available
import numpy as np

from CAT.strategy.abstract_strategy import AbstractStrategy
from CAT.model import AbstractModel
from CAT.dataset import AdapTestDataset


class RandomStrategy(AbstractStrategy):

    def __init__(self):
        super().__init__()

    @property
    def name(self):
        return 'Random Select Strategy'

    def adaptest_select(self, model: AbstractModel, sid, adaptest_data: AdapTestDataset, item_candidates=None):
        if item_candidates is None:
            untested_questions = np.array(list(adaptest_data.untested[sid]))
        else:
            available = adaptest_data.untested[sid].intersection(set(item_candidates))
            untested_questions = np.array(list(available))
        # pred_all = model.get_pred(adaptest_data)
        # fuzzy_order = sorted([(abs(pred_all[sid][qid]-0.5), qid) for qid in untested_questions])
        # untested_questions = [ q for _, q in fuzzy_order[:100]]
        return untested_questions[np.random.randint(len(untested_questions))]