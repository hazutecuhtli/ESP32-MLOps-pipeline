import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def Ridge_Regression_Results(df_metrics, y_test_kit, y_pred_kit, y_pred_ridge_kit):
    
    fig = plt.figure(figsize=(11,4), facecolor='lightgray')
    fig.subplots_adjust(left=0.2,wspace = 0.1)
    ax1 = plt.subplot2grid((1,2),(0,0)) 
    bbox=[0, 0, 1, 1]
    ax1.axis('off')
    table = ax1.table(cellText=df_metrics.values, bbox=[0,0,1,.7], colLabels=df_metrics.columns, colWidths=[.07,.1,.1,.1,.1])
    table.set_fontsize(14)
    table.scale(3, 1)
    ax1.set_title('Comparison of Linear and Ridge Regressors')
    for (row,col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight='bold', horizontalalignment='right')
        if (row == 0) & (col==0):
            cell.set_facecolor('white')
            cell.set_edgecolor('white')    
        elif (row == 0) & (col==1):
            cell.set_text_props(weight='bold', horizontalalignment='right')
            cell.set_facecolor('white')
            cell.set_edgecolor('white')
        elif (row%2 == 0):
            if  col == 0:
                cell.set_text_props(weight='bold', horizontalalignment='center', fontsize=32)
            cell.set_edgecolor('white')
            cell.set_facecolor('white')
        elif (row%2 == 1):
            if col == 0:
                cell.set_text_props(weight='bold', horizontalalignment='center')
            cell.set_edgecolor('white')
            cell.set_facecolor('whitesmoke')        
    ax2 = plt.subplot2grid((1,2),(0,1),colspan=1) 
    ax2.plot(y_test_kit, label='Real Value', color='black', linewidth=1.5)
    ax2.plot(y_pred_kit, label='Prediction', color='royalblue', linestyle='--')
    ax2.plot(y_pred_ridge_kit, label='Ridge Reg.', color='red', linestyle='-')
    ax2.set_title("Kitchen Temperature – Predictions")
    ax2.set_xlabel("Sliding Window")
    ax2.set_ylabel("Temperature [°C]")
    ax2.legend(loc='upper right')
    ax2.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()