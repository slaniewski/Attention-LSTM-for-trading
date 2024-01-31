from src.init_setup import *
import sys
import os
import pickle
import shutil
import numpy as np
from collections import Counter
import json
import time


class PerformanceMetrics:
    def __init__(self) -> None:
        self.setup = Setup()
        self.config = self.setup.config
        self.logger = logging.getLogger("Performance Metrics")
        self.logger.addHandler(logging.StreamHandler())
        self.eval_data = None
        self.eval_data_timestamp = None
        self.pred_thres = float(self.config["model"]["PredictionThreshold"])
        self.tr_cost = float(self.config["evaluation"]["TransactionCost"])
        self.export_path = ""

    def calculate_metrics(self, custom_timestamp=None):
        """
        Calculate ARC (for buy-and-hold strategy and LSTM strategy), ASD, IR, MLD (only for LSTM strategy)
        :return: equity line array for further visualization
        """

        # Load evaluation data for performance metrics
        self.load_latest_eval_data(custom_timestamp)
        returns_array, inside_thres_count, positions_array, transactions = (
            self.returns(self.eval_data["Pred"].values, self.eval_data["Real"].values)
        )
        self.logger.info(
            f'Out of {returns_array.shape[0]} observations, there were {inside_thres_count} '
            + f'predictions under threshold of {self.pred_thres}')
        self.logger.info(
            f'There were {transactions} transactions')
        pos_counter = Counter(positions_array)
        self.logger.info(f'Positions: [Long]: {pos_counter[1]}, [Short]: {pos_counter[-1]}')
        equity_line = self.eq_line(returns_array, self.eval_data["Real"].values[0])
        accuracy_val = self.accuracy(self.eval_data["Pred"].values, self.eval_data["Real"].values)
        # Save performance statistics to a dictionary
        metrics = {
            "[ARC_BH]": str(np.round(self.arc(self.eval_data["Real"].values) * 100, 2)) + "%",
            "[ARC_EQ]": str(np.round(self.arc(equity_line) * 100, 2)) + "%",
            "[ASD_EQ]": str(np.round(self.asd(equity_line), 4)),
            "[IR_EQ]": str(np.round(self.ir(equity_line), 4)),
            "[Accuracy]": str(np.round(accuracy_val, 2)) + "%",
            "[Days_to_new_peak]": str(np.round(self.days_to_new_peak(returns_array), 2)),
            "[Loss_in_row]": str(np.round(self.mld(returns_array), 2)) # + "% of the year"
        }

        # Save results to .json file
        performance_metrics_path = f'{self.setup.ROOT_PATH}{self.config["prep"]["ModelMetricsDir"]}performance_metrics_{self.eval_data_timestamp}.json'
        with open(performance_metrics_path, 'w') as fp:
            json.dump(metrics, fp, indent=4, sort_keys=False)
        shutil.copy2(performance_metrics_path, self.export_path)
        eq_line_path = f'{self.setup.ROOT_PATH}{self.config["prep"]["DataOutputDir"]}eq_line_{self.eval_data_timestamp}.pkl'
        self.logger.info(f'Saving Equity Line array: {eq_line_path}')
        with open(eq_line_path, 'wb') as handle:
            pickle.dump(np.asarray(equity_line_array), handle, protocol=pickle.HIGHEST_PROTOCOL)
        shutil.copy2(eq_line_path, self.export_path)

        self.logger.info(f'[ARC_BH]:    {metrics["[ARC_BH]"]}')
        self.logger.info(f'[ARC_EQ]:    {metrics["[ARC_EQ]"]}')
        self.logger.info(f'[ASD_EQ]:    {metrics["[ASD_EQ]"]}')
        self.logger.info(f'[IR_EQ]:     {metrics["[IR_EQ]"]}')
        self.logger.info(f'[Accuracy]:     {metrics["[Accuracy]"]}') 
        self.logger.info(f'[Days_to_new_peak]:     {metrics["[Days_to_new_peak]"]}') 
        self.logger.info(f'[Loss_in_row]:    {metrics["[Loss_in_row]"]}')

        return equity_line

    # tutaj zrobic loop po kazdym z modeli, a nie brac tylko ostatni
    def load_eval_data(self, custom_timestamp = None) -> int:
        files = os.listdir(self.config["prep"]["DataOutputDir"])
        
        if custom_timestamp is None:
            pickles = [f'{self.config["prep"]["DataOutputDir"]}{f}' for f in files if ("model_eval_data" in f and f.endswith(".pkl"))]
            try: latest_file = max(pickles, key=os.path.getmtime)
            except ValueError as ve:
                self.logger.error("No file available. Please rerun the whole process / load data first.")
                sys.exit(1)
            self.logger.info(f"Found latest eval data pickle: {latest_file}")
            self.eval_data_timestamp = latest_file[-20:-4]
        else: 
            self.eval_data_timestamp = custom_timestamp
            custom_eval_data_path = f'{self.config["prep"]["DataOutputDir"]}model_eval_data_{custom_timestamp}.pkl'
            self.logger.debug(f'Trying to load eval data: {custom_eval_data_path}')
            try: 
                with open(custom_eval_data_path, 'rb') as handle: self.eval_data = pickle.load(handle)
            except FileNotFoundError as ve: 
                self.logger.error("Model evaluation data not found.")
                sys.exit(1)
            self.logger.info(f"Loaded model evaluation data pickle: {custom_eval_data_path}")

        self.export_path = f'{self.setup.ROOT_PATH}{self.config["prep"]["ExportDir"]}{self.eval_data_timestamp}/'
        if not os.path.isdir(self.export_path): os.mkdir(self.export_path)

        return 0

    def equity_line(self, predictions: np.array, actual_values: np.array) -> tuple:
        """
        Calculate returns from investment based on predictions for price change, and actual values
        used params:
            - threshold required to consider a prediction as reliable or not (0 by default)
            - transaction cost of changing the investment position. Counts double.

        table of possible cases:
        | previous | current | threshold    | decision   |
        |----------|---------|--------------|------------|
        | L        | L       | abs(x)>thres | L (keep)   |
        | L        | L       | abs(x)<thres | L (keep)   |
        | L        | S       | abs(x)>thres | S (change) |
        | L        | S       | abs(x)<thres | L (keep)   |
        | S        | L       | abs(x)>thres | L (change) |
        | S        | L       | abs(x)<thres | S (keep)   |
        | S        | S       | abs(x)>thres | S (keep)   |
        | S        | S       | abs(x)<thres | S (keep)   |

        :param predictions: array of values between [-1, 1]
        :param actual_values: array of actual values
        :return: returns array, counter of transactions below threshold, array of position indicators
        """
        positions = [-1]  # store positions (1 for long, -1 for short), first is short by default
        transactions, counter = 0, 0  # count positions inside the threshold
        returns_array = []  # output array
        start_of_prediction = 1 #change for last datapoint in train for TEST DATASET
        for i in range(start_of_prediction, actual_values.shape[0], 1):  # for i in actual values

            # If Previous long
            if positions[i-1] == 1:

                # If current long => threshold doesn't matter, keep long
                if predictions[i] > 0:
                    returns_array.append(actual_values[i] - actual_values[i - 1])
                    positions.append(1)

                # If current short => check threshold
                else: #predictions[i] < 0:

                    # If abs(x) > threshold => position change long->short
                    if abs(predictions[i]) > self.pred_thres:
                        returns_array.append(actual_values[i - 1] - actual_values[i] - 2 * self.tr_cost * actual_values[i - 1])
                        positions.append(-1)
                        transactions += 1

                    # If abs(x) < threshold => long position unchanged
                    else: # abs(predictions[i]) < self.pred_thres:
                        returns_array.append(actual_values[i] - actual_values[i - 1])
                        positions.append(1)
                        counter += 1

            # If Previous short
            else: #if positions[i-1] == -1:
                
                # If current short => threshold doesn't matter, keep short
                if predictions[i] < 0:
                    returns_array.append(actual_values[i - 1] - actual_values[i])
                    positions.append(-1)

                # If current long => check threshold
                else: #if predictions[i] > 0:

                    # If abs(x) > threshold => position change short->long
                    if abs(predictions[i]) > self.pred_thres:
                        returns_array.append(actual_values[i] - actual_values[i - 1] - 2 * self.tr_cost * actual_values[i - 1])
                        positions.append(1)
                        transactions += 1

                    # If abs(x) < threshold => short position unchanged
                    else: #if abs(predictions[i]) < self.pred_thres:
                        returns_array.append(actual_values[i - 1] - actual_values[i])
                        positions.append(-1)
                        counter += 1

        return np.asarray(returns_array), counter, positions, transactions

    def eq_line(self, returns_array, _n_value):
        """
        Calculate the equity line of investment
        Equity Line informs about the change of accumulated capital
        :param returns_array: array of returns from investment
        :param _n_value: Initial value of capital
        :return: returns an array where each element represents the capital at the given timestep
        """

        # first value is the initial value of the capital
        equity_array = [_n_value]

        # Walk through every return in returns array
        # For each daily investment return, add (the current equity value + daily change) to the array
        for i, x in enumerate(returns_array):
            equity_array.append(equity_array[i] + x)

        # Save Equity Line array for visualization module
        output_dir = self.setup.ROOT_PATH + self.config["prep"]["DataOutputDir"]
        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)
        eq_line_path = f'{self.config["prep"]["DataOutputDir"]}eq_line_{self.eval_data_timestamp}.pkl'
        self.logger.info(f'Saving Equity Line array: {eq_line_path}')
        with open(eq_line_path, 'wb') as handle:
            pickle.dump(np.asarray(equity_array), handle, protocol=pickle.HIGHEST_PROTOCOL)

        return np.asarray(equity_array)

    def asd(self, equity_array, scale=252):
        """
        Annualized Standard Deviation
        :param equity_array: array of investment return for each timestep
        :param scale: number of days required for normalization. By default, in a year there are 252 trading days.
        :return: ASD as percentage
        """

        # Differentiate the returns array to get percentages instead of nominal values
        return_diffs = self.diff_array(equity_array)

        return np.std(return_diffs) * np.sqrt(scale)

    @staticmethod
    def diff_array(values_array: np.array, cost: float = 0) -> np.array:
        """
        Calculates discrete differences as percentages of the previous value
        :param values_array: Real prices array.
        :param cost: Constant cost for each timestep
        :return: Differentiated array
        """
        results_array = []
        for i in range(values_array.shape[0]):
            if i == 0:
                continue
            else:
                results_array.append((values_array[i] - values_array[i-1]) / values_array[i-1] - cost)
        return np.asarray(results_array)

    @staticmethod
    def arc(equity_array: np.array, scale: int = 252) -> float:
        """
        Annualized Return Ratio
        :param equity_array: equity line
        :param scale: number of days required for normalization. By default in a year there are 252 trading days.
        :return: ARC as percentage
        """
        ratio = equity_array[-1] / equity_array[0]
        result = np.sign(ratio) * (np.abs(ratio) ** (scale / len(equity_array))) - 1
        return result

        # return (equity_array[-1] / equity_array[0]) ** (scale / len(equity_array)) - 1

    def ir(self, equity_array: np.array, scale=252) -> float:
        """
        Information Ratio
        :param equity_array: Equity Line array
        :param equity_array: array of investment return for each timestep
        :param scale: number of days required for normalization. By default in a year there are 252 trading days.
        """
        return self.arc(equity_array, scale) / self.asd(equity_array, scale)

    @staticmethod
    def mld(returns_array: np.array, scale: int = 1) -> float:
        """
        Maximum Loss Duration (MLD)
        Calculates the maximum number of consecutive time steps when the returns were below 0.
        
        :param returns_array: Array of investment returns.
        :param scale: Number of days required for normalization. By default, in a year there are 252 trading days.
        :return: MLD
        """
        
        max_consecutive_losses = 0
        current_consecutive_losses = 0
        
        for ret in returns_array:
            if ret < 0:
                current_consecutive_losses += 1
                max_consecutive_losses = max(max_consecutive_losses, current_consecutive_losses)
            else:
                current_consecutive_losses = 0

        # Normalize over the number of trading days in a year
        return max_consecutive_losses / scale
    
    @staticmethod
    def days_to_new_peak(returns_array: np.array) -> int:
        """
        Days to New Peak (DNP)
        Calculates the maximum number of consecutive days until a new peak in returns was achieved.
        
        :param returns_array: Array of cumulative investment returns.
        :return: DNP (Days to New Peak)
        """
        
        max_days_to_peak = 0
        current_days_to_peak = 0
        current_peak = returns_array[0]

        for ret in returns_array:
            if ret > current_peak:
                current_peak = ret
                current_days_to_peak = 0
            else:
                current_days_to_peak += 1
                max_days_to_peak = max(max_days_to_peak, current_days_to_peak)

        return max_days_to_peak

    
    @staticmethod
    def accuracy(predicted_values: np.array, actual_values: np.array) -> float:
        """
        Calculates the accuracy for binary predictions.
        
        :param predicted_values: Array of predicted continuous values.
        :param actual_values: Array of actual continuous values.
        :return: Accuracy as a percentage.
        """

        # Convert continuous values to binary labels
        predicted_labels = np.where(np.diff(predicted_values) >= 0, 1, 0)
        actual_labels = np.where(np.diff(actual_values) >= 0, 1, 0)

        # Calculate the number of correct predictions
        correct_predictions = np.sum(predicted_labels == actual_labels)

        # Calculate accuracy
        accuracy_percentage = (correct_predictions / len(predicted_labels)) * 100

        return accuracy_percentage
