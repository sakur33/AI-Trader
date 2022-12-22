import glob
import os
import pickle as pkl
from sklearn.metrics import accuracy_score
from trading_accounts import trader
import time
from utils import (
    get_today,
    generate_clustered_dataset,
    build_lstm_v1,
    train_model,
    get_test_df,
    get_last_step,
    calculate_stop_loss,
    calculate_take_profit,
    plot_stock,
)

today = get_today()
curr_path = os.path.dirname(os.path.realpath(__file__))
data_path = curr_path + "../../data/"
symbol_path = curr_path + "../../symbols/"
cluster_path = curr_path + "../../clusters/"
model_path = curr_path + "../../model/"
result_path = curr_path + "../../result/"
docs_path = curr_path + "../../docs/"


# trader1 = trader(capital=1000, max_risk=0.1, trade_type="STC")
# trader1.make_trade("", 100)

today = get_today()
pre_step = 5
steps = 30
stop_step = 5
pred_threshold = 0.5
CURRENT_PRICE = 100


picks = glob.glob(f"{data_path}*.pickle")
symbols = []
for pick in picks:
    symbols.append(picks[0].split("\\")[-1].split("_")[0])

print("--------------------------------------------------------")
dataset_dict, sc = generate_clustered_dataset(
    picks, symbols, steps=steps, pred_step=pre_step
)
model = build_lstm_v1(
    128, [dataset_dict["X_train"].shape[1], dataset_dict["X_train"].shape[2]]
)

model, history, test_preds = train_model(
    model,
    dataset_dict["X_train"],
    dataset_dict["X_val"],
    dataset_dict["X_test"],
    dataset_dict["y_train"],
    dataset_dict["y_val"],
    dataset_dict["y_test"],
    epochs=100,
    patience=15,
)
shape_x_train = dataset_dict["X_train"].shape
shape_x_val = dataset_dict["X_val"].shape
shape_x_test = dataset_dict["X_test"].shape
shape_y_train = dataset_dict["y_train"].shape
shape_y_val = dataset_dict["y_val"].shape
shape_y_test = dataset_dict["y_test"].shape
print(
    f"Shapes: {shape_x_train} | {shape_x_val} | {shape_x_test} | {shape_y_train} | {shape_y_val} | {shape_y_test}"
)
if test_preds > 0.5:
    for pick in picks:

        test_df, real_df, y_pred = get_test_df(pick, model, sc, steps)
        test_df["y_pred"][test_df["y_pred"] > pred_threshold] = 1
        test_df["y_pred"][test_df["y_pred"] < pred_threshold] = 0
        symbol = picks[0].split("\\")[-1].split("_")[0]
        acc = accuracy_score(test_df["y_test"], test_df["y_pred"])
        print(f"{symbol} | Acc: {acc}")

        if acc > 0.7:
            last_step = get_last_step(real_df, sc, steps)
            prediction = model.predict(last_step.reshape(1, steps, -1))[0]
            if prediction < pred_threshold:
                short = True
                trans_txt = "SELL"
            else:
                short = False
                trans_txt = "BUY"

            buy_price = real_df["close"].values[-1]
            stop_loss = calculate_stop_loss(real_df, stop_step, short)
            take_profit = calculate_take_profit(buy_price, stop_loss, short)

            print(
                f"{trans_txt} Transaction on {symbol} -> BP:{buy_price} | SL:{stop_loss} | TP: {take_profit} || Prediction of {prediction}"
            )
            plot_stock(
                df=real_df,
                symbols=[symbol],
                params={
                    "buy_price": buy_price,
                    "stop_loss": stop_loss,
                    "take_profit": take_profit,
                },
            )
            time.sleep(0.5)
        else:
            plot_stock(
                df=real_df,
                symbols=[symbol],
                params=None,
            )
