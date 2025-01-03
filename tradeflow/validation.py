import numpy as np
import tradeflow
from tradeflow.common.logger_utils import get_logger
from pathlib import Path
import time

DATA_FOLDER = Path(__file__).parent.joinpath("data")
logger = get_logger(__file__)

signs = np.loadtxt(DATA_FOLDER.joinpath("signs-20240720.txt"), dtype="int8", delimiter=",")

max_orders = [20, 60]
fit_methods = ["yule_walker", "burg", "ols_with_cst"]
seeds = [1, None]

s1 = time.time()
for max_order in max_orders:
    for fit_method in fit_methods:
        for seed in seeds:
            logger.info(f"START")
            logger.info(f"max_order: {max_order} | order_selection_method: pacf | fit_method: {fit_method} | seed: {seed}")
            s2 = time.time()
            ar_model = tradeflow.AR(signs=signs, max_order=max_order, order_selection_method="pacf")
            ar_model = ar_model.fit(method=fit_method, significance_level=0.05, check_residuals=True)

            ar_model.simulate(size=300_000, seed=seed)

            simulation_summary = ar_model.simulation_summary(plot=True, log_scale=True)
            logger.info(f"\n{simulation_summary}")
            e2 = time.time()
            logger.info(f"END | =====> TIME: {round(e2 - s2, 4)}s\n\n")

e1 = time.time()
logger.info(f"!!!! TOTAL TIME: {round(e1 - s1, 4)}s !!!!")
fig = ar_model.plot_acf_and_pacf(nb_lags=52, log_scale=True)
