import glob
import os
import pickle as pkl
from sklearn.metrics import accuracy_score
from utils import (
    get_today,
    generate_clustered_dataset,
    build_lstm_v1,
    train_model,
    get_test_df,
    get_last_step,
    calculate_stop_loss,
    calculate_take_profit,
)

today = get_today()
curr_path = os.path.dirname(os.path.realpath(__file__))
data_path = curr_path + "../../data/"
symbol_path = curr_path + "../../symbols/"
cluster_path = curr_path + "../../clusters/"
model_path = curr_path + "../../model/"
docs_path = curr_path + "../../docs/"


today = get_today()
step = 10
pred_threshold = 0.5
CURRENT_PRICE = 100

with open(f"{cluster_path}grouper_" + today + ".pickle", "rb") as f:
    group_dict = pkl.load(f)

picks = glob.glob(f"{data_path}*.pickle")
for group in group_dict["Clusters"]:
    print("--------------------------------------------------------")
    print(f"GROUP: {group}")
    dataset_dict, sc = generate_clustered_dataset(picks, group)

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

    if test_preds > 0.5:
        for pick in picks:
            if any(symbol in pick for symbol in group_dict["Clusters"][0]):
                test_df, real_df, y_pred = get_test_df(pick, model)
                test_df["y_pred"][test_df["y_pred"] > pred_threshold] = 1
                test_df["y_pred"][test_df["y_pred"] < pred_threshold] = 0
                symbol = pick.split("_")[0].split("\\")[-1]
                acc = accuracy_score(test_df["y_test"], test_df["y_pred"])
                print(f"{symbol} | Acc: {acc}")

                if acc > 0.85:
                    last_step = get_last_step(real_df, sc, step)
                    prediction = model.predict(last_step.reshape(1, step, -1))[0]
                    if prediction < pred_threshold:
                        short = True
                        trans_txt = "SELL"
                    else:
                        short = False
                        trans_txt = "BUY"

                    buy_price = real_df["close"].values[-1]
                    stop_loss = calculate_stop_loss(real_df, step, short)
                    take_profit = calculate_take_profit(buy_price, stop_loss, short)
                    print(
                        f"{trans_txt} Transaction on {symbol} -> BP:{buy_price} | SL:{stop_loss} | TP: {take_profit} || Prediction of {prediction}"
                    )
                    input()
                    # plot_stock(
                    #     real_df,
                    #     {
                    #         "buy_price": buy_price,
                    #         "stop_loss": stop_loss,
                    #         "take_profit": take_profit,
                    #     },
                    #     symbol,
                    # )
