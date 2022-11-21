from st_aggrid import AgGrid, GridUpdateMode, DataReturnMode
from st_aggrid.grid_options_builder import GridOptionsBuilder
import joblib
import lightgbm
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os.path
# print(os.path.dirname(__file__))
# current  = os.path.dirname(__file__)
# parent = os.path.split(current)[0]
# print(parent)
# model_path = os.path.join(parent,'model')
# print(model_path)
st.set_page_config(layout="wide")



model = lightgbm.Booster(model_file='lgbr_base.txt')
st.write("RUL Alert Project")
csv=st.file_uploader("Upload CSV File")
if csv is not None:
    df = pd.read_csv(csv)
    # st.write(df)

    df_filtered = df.filter(['EngineNo', 'Cycle', 'OpSet1', 'OpSet2', 'LPCOutletTemp', 'HPCOutletTemp', 'LPTOutletTemp',
                             'TotalHPCOutletPressure', 'PhysicalFanSpeed', 'PhysicalCoreSpeed',
                             'StaticHPCOutletPressure',
                             'FuelFlowRatio', 'CorctFanSpeed', 'CorctCoreSpeed', 'BPR', 'HPTCoolantBleed',
                             'LPTCoolantBleed'], axis=1)
    scaler = joblib.load("scaler.save")
    names = df_filtered.iloc[:, 2:].columns
    scaled_data = scaler.transform(df_filtered.iloc[:, 2:])
    df_test_scaled = pd.concat([df_filtered.iloc[:, 0:2], pd.DataFrame(scaled_data, columns=names)], axis=1)
    # print(df_test_scaled)
    X_pred = df_test_scaled.drop(['EngineNo', 'Cycle'], axis=1)
    y_pred_temp = model.predict(X_pred)
    y_pred = (y_pred_temp >= 0.5).astype('int')
    df["result"]= y_pred

    df.drop(df.filter(regex="Unname"), axis=1, inplace=True)
    df_tmp = df.copy()
    # st.write(df_tmp)

    # fig, ax = plt.subplots()
    # ax = df_tmp.plot(x="EngineNo", y=['Cycle', 'OpSet1', 'OpSet2', 'LPCOutletTemp', 'HPCOutletTemp', 'LPTOutletTemp',
    #                          'TotalHPCOutletPressure', 'PhysicalFanSpeed', 'PhysicalCoreSpeed',
    #                          'StaticHPCOutletPressure',
    #                          'FuelFlowRatio', 'CorctFanSpeed', 'CorctCoreSpeed', 'BPR', 'HPTCoolantBleed',
    #                          'LPTCoolantBleed'], kind="bar", rot=0)
    # st.pyplot(ax.figure)
    csv_new = df.to_csv().encode('utf-8')
    st.download_button(
        "Download Result",
        csv_new,
        "file.csv",
        "text/csv",
        key='download-csv'
    )

    # st.bar_chart(data=df_tmp.T)

    def display_engines_under_rul():
        list_of_engines_under_rul = df_cmp[(df_cmp.result == 1)]['EngineNo'].unique()
        st.write("Engines that need maintenance: ")
        list_of_engines = ''
        for a in list_of_engines_under_rul:
            list_of_engines = list_of_engines+str(a)+', '
        list_of_engines = list_of_engines[:-2]
        st.write(list_of_engines)

    def create_grid(df_grid):
        gd = GridOptionsBuilder.from_dataframe(df_grid)
        gd.configure_default_column(enablePivot=True, enableValue=True, enableRowGroup=True) #1
        gd.configure_selection(selection_mode='multiple', use_checkbox=True)
        gd.configure_side_bar()
        gridoptions = gd.build()
        customcss = {"#gridContainer":{"width":"1000px !important"}}
        left, right = st.columns([3,1])
        with left:
            grid_table = AgGrid(df_grid, height=250, width=50, gridOptions=gridoptions,
                                update_mode=GridUpdateMode.SELECTION_CHANGED, enable_enterprise_modules=True,
                                data_return_mode=DataReturnMode.FILTERED_AND_SORTED,
                                fit_columns_on_grid_load=False,
                                header_checkbox_selection_filtered_only=True,
                                custom_css=customcss,
                                use_checkbox=True)
        with right:
            display_engines_under_rul()


        # st.write('## Selected')
        selected_row = grid_table["selected_rows"]
        # st.dataframe(selected_row)
        selected_data = pd.DataFrame(selected_row)
        selected_data.drop(selected_data.filter(regex="_selectedRowNodeInfo"), axis=1, inplace=True)

        st.write(selected_data)
        st.bar_chart(data=selected_data.T)


    df_cmp = df_tmp.copy()
    create_grid(df_cmp)


    # st.write('## Filter records with RUL under 30')
    # create_grid(df_cmp[(df_cmp.result == 1)])
