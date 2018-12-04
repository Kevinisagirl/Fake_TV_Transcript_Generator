import plotly.plotly as py
import cufflinks as cf
import pandas as pd
import numpy as np
from plotly.offline import download_plotlyjs,init_notebook_mode,plot,iplot
cf.go_offline()

def make_plot(df):
    fig = df.iplot(asFigure=True, xTitle='SBIR Award Year', yTitle='Document Counts', title='Count of Documents in each Topic over Time')
    plot(fig, filename='lda_plot_SBIR.html')