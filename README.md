See the full description here: [https://evgenypogorelov.com/portfolio-rebalancing-python](https://evgenypogorelov.com/portfolio-rebalancing-python)

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/pogoetic/rebalance/master)

Timely and consistent rebalancing is a cornerstone of modern portfolio theory. Rebalancing can magnify returns by promoting selling high and buying low, and reduce long-term risk by ensuring the portfolio adheres to its designated risk tolerance. The keys are to rebalance in a timely manner (i.e. annually) and to do it consistently because the benefits of rebalancing compound over time - but rebalancing by hand is a pain, and that can lead to inconsistency. I pursued this project due to the lack of free tools to simply rebalance an existing portfolio. Sure, we can all use a worksheet to do the math each time, but why not automate it and make it more likely we will actually do it? Here, inspired by the work of [kdboller](https://nbviewer.jupyter.org/github/kdboller/pythonsp500/blob/a7066d998ff046c3cc8b26ece3b0efdf00959d57/Investment%20Portfolio%20Python%20Notebook_03_2018_blog%20example.ipynb) I'll use Pandas, the Tiingo API, and some simple math to calculate how to optimally rebalance a portfolio given a target allocation. This is a simple, no-frills portfolio rebalancing exercise which does not factor in important considerations such as tax efficiency, transaction costs, or alternate approaches such as stock-out rebalances, or bond-floor settings. Future versions of this project may contemplate these extra factors.  


**Steps:** 
1. Set triggers to rebalance (time or threshold or both)
2. Define our current Portfolio (accounttype, time, ticker, shares, cost basis, price)  
3. Define our target allocation (ticker, allocation)  
4. Factor in any new money being invested  
5. Calculate initial transactions needed to hit target allocation  
6. Determine which transactions are valid based on rebalance triggers
7. Iteratively determine sells and buys required to get as close as possible to target allocation  


**References:**

[pythonsp500 by kdboller](https://nbviewer.jupyter.org/github/kdboller/pythonsp500/blob/a7066d998ff046c3cc8b26ece3b0efdf00959d57/Investment%20Portfolio%20Python%20Notebook_03_2018_blog%20example.ipynb)

[Portfolio Rebalancing by bogleheads wiki](https://www.bogleheads.org/wiki/Rebalancing)

[The Rebalancing Effect by Morgan Stanely](https://www.morganstanley.com/articles/rebalancing-effect)

