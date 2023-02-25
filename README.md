# AI based trader for xtb [ON THE WORKS]

This project sets up a automatical python-based stock trader.  
The overal workflow for this system is separated in two different processes that ought to be launched in an automated way:

- Offline process
- Online process

Aside from the actual working processess, an analytical tool has been also developed.

## Offline Process

The offline process contemplates the adjustment and optimization of the trading parameters needed to run the online process of the system. This process is launched automatically in two ways, when the market is closed and when the current parameters are underperforming.

Within the offline process we have 4 different steps that are taken:

- Current broker's symbols are gathered and the DB is updated.
- All current symbols are analyzed in search for suitable assests
- Selected assest's data is gathered for the last 7 days
- The trading logic is optimized given the retrieved data
- The trading parameters for the well performing assets is stored in the DB

## Online Process

The online process contemplates the start, execution and ending of a trading session. This means that whenever an updated trading param is included in the database and the market is open, the system will automatically launch a trading session usign this parameters.

Within this online process a few steps are taken:

- The current parameter db is inspected and the most updated parameters are retrieved.
- If the parameters fit the curret date, a trading session will be launched
- Within a trading session, the trading logic will be tested and applied
- Under the following conditions the trading session will end:
  - If the parameters underperform
  - If the market is about to close
  - If the session capital is lost

## Monitoring Dashboard

This dashboard is intended to monitor the performance of the system. It contains a dashboard ment to visualize the following:

- A graph showing the candles of the asset
- A graph showing the session status through the timeline (in trade, off trade)
- A graph showing the evolution of the ask/bid and volumes
- A gauge display showing the profits and trades made
- A bar graph showing the stats of the trading session
