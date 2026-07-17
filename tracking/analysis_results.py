import _init_paths
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [8, 8]

from lib.test.analysis.plot_results import plot_results, print_results, print_per_sequence_results
from lib.test.evaluation import get_dataset, trackerlist

trackers = []
dataset_name = 'nfs' # choosen from 'uav', 'nfs', 'lasot_extension_subset', 'lasot', 'lasot_lang', 'otb99_lang', 'tnl2k'

trackers.extend(trackerlist(name='flextrackv2', parameter_name='flextrackv2_t224', dataset_name=dataset_name,
                            run_ids=None, display_name='flextrackv2'))

dataset = get_dataset(dataset_name)

print_results(trackers, dataset, dataset_name, merge_results=True, plot_types=('success', 'prec', 'norm_prec'),
              force_evaluation=True)

