from st_aggrid import AgGrid, GridUpdateMode, DataReturnMode
from st_aggrid.grid_options_builder import GridOptionsBuilder
import joblib
import lightgbm
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os.path


st.set_page_config(layout="wide", page_title="RUL Predictor")
model = lightgbm.Booster(model_file='lgbr_base.txt')
scaler = joblib.load("scaler.save")

# st.header("Predict RUL")
st.markdown("<h1 style='text-align: center; color: grey;'>Predict RUL</h1>", unsafe_allow_html=True)


# def scroll():
#     st.markdown("[Upload File](#upload-file)", unsafe_allow_html=True)
#
# st.button("scroll down", on_click=scroll)
# st.markdown("[Upload File](#upload-file)", unsafe_allow_html=True)
st.markdown("<a style='color: grey;text-decoration: none;border-style: outset;border-radius: 2px;' href='#upload-file'>Upload File</a>", unsafe_allow_html=True)
option = st.selectbox('',('Select','Engine1', 'Engine2'))
if option == 'Engine1':
    st.session_state["OpSet1"] = "300"
    st.session_state["OpSet2"] = "493"
    st.session_state["LPCOutletTemp"] = "484"
    st.session_state["HPCOutletTemp"] = "422"
    st.session_state["LPTOutletTemp"] = "945"
    st.session_state["TotalHPCOutletPressure"] = "482"
    st.session_state["PhysicalFanSpeed"] = "383"
    st.session_state["PhysicalCoreSpeed"] = "483"
    st.session_state["StaticHPCOutletPressure"] = "858"
    st.session_state["FuelFlowRatio"] = "492"
    st.session_state["CorctFanSpeed"] = "438"
    st.session_state["CorctCoreSpeed"] = "834"
    st.session_state["BPR"] = "399"
    st.session_state["HPTCoolantBleed"] = "483"
    st.session_state["LPTCoolantBleed"] = "383"
elif option == 'Engine2':
    st.session_state["OpSet1"] = "0.001"
    st.session_state["OpSet2"] = "0.22"
    st.session_state["LPCOutletTemp"] = "24"
    st.session_state["HPCOutletTemp"] = "21"
    st.session_state["LPTOutletTemp"] = "23"
    st.session_state["TotalHPCOutletPressure"] = "42"
    st.session_state["PhysicalFanSpeed"] = "21"
    st.session_state["PhysicalCoreSpeed"] = "21"
    st.session_state["StaticHPCOutletPressure"] = "23"
    st.session_state["FuelFlowRatio"] = "34"
    st.session_state["CorctFanSpeed"] = "34"
    st.session_state["CorctCoreSpeed"] = "22"
    st.session_state["BPR"] = "32"
    st.session_state["HPTCoolantBleed"] = "33"
    st.session_state["LPTCoolantBleed"] = "23"


# st.write('## Filter records with RUL under 30')
# create_grid(df_cmp[(df_cmp.result == 1)])
form = st.form("my_form")
form.write("Enter sensory values to check the status of the engine")
left,center, right = form.columns([1, 1, 1])

with left:
    OpSet1 = st.text_input('Operational Setting 1', key='OpSet1', placeholder="Operational Setting 1",label_visibility="visible")
    HPCOutletTemp = st.text_input('HPC Outlet Temperature', key='HPCOutletTemp', placeholder="HPC Outlet Temperature",label_visibility="visible")
    PhysicalFanSpeed = st.text_input('Physical Fan Speed', key='PhysicalFanSpeed', placeholder="Physical Fan Speed",label_visibility="visible")
    FuelFlowRatio = st.text_input('Fuel Flow Ratio', key='FuelFlowRatio', placeholder="Fuel Flow Ratio",label_visibility="visible")
    BPR = st.text_input('BPR', key='BPR', placeholder="BPR",label_visibility="visible")
with center:
    OpSet2 = st.text_input('Operational Setting 2', key='OpSet2', placeholder="Operational Setting 2",label_visibility="visible")
    LPTOutletTemp = st.text_input('LPT Outlet Temperature', key='LPTOutletTemp', placeholder="LPT Outlet Temperature",label_visibility="visible")
    PhysicalCoreSpeed = st.text_input('Physical Core Speed', key='PhysicalCoreSpeed', placeholder="Physical Core Speed",label_visibility="visible")
    CorctFanSpeed = st.text_input('Corct Fan Speed', key='CorctFanSpeed', placeholder="Corct Fan Speed",label_visibility="visible")
    LPTCoolantBleed = st.text_input('LPT Coolant Bleed', key='LPTCoolantBleed', placeholder="LPT Coolant Bleed",label_visibility="visible")
with right:
    # engine_no = st.text_input('', key='engine_no', placeholder="Engine Number")
    LPCOutletTemp = st.text_input('LPC Outlet Temperature', key='LPCOutletTemp', placeholder="LPC Outlet Temperature",label_visibility="visible")
    TotalHPCOutletPressure = st.text_input('Total HPC Outlet Pressure', key='TotalHPCOutletPressure', placeholder="Total HPC Outlet Pressure",label_visibility="visible")
    StaticHPCOutletPressure = st.text_input('Static HPC Outlet Pressure', key='StaticHPCOutletPressure', placeholder="Static HPC Outlet Pressure",label_visibility="visible")
    CorctCoreSpeed = st.text_input('Corct Core Speed', key='CorctCoreSpeed', placeholder="Corct Core Speed",label_visibility="visible")
    HPTCoolantBleed = st.text_input('HPT Coolant Bleed', key='HPTCoolantBleed', placeholder="HPT Coolant Bleed",label_visibility="visible")


def clear_form():
    st.session_state["OpSet1"] = ""
    st.session_state["OpSet2"] = ""
    st.session_state["LPCOutletTemp"] = ""
    st.session_state["HPCOutletTemp"] = ""
    st.session_state["LPTOutletTemp"] = ""
    st.session_state["TotalHPCOutletPressure"] = ""
    st.session_state["PhysicalFanSpeed"] = ""
    st.session_state["PhysicalCoreSpeed"] = ""
    st.session_state["StaticHPCOutletPressure"] = ""
    st.session_state["FuelFlowRatio"] = ""
    st.session_state["CorctFanSpeed"] = ""
    st.session_state["CorctCoreSpeed"] = ""
    st.session_state["BPR"] = ""
    st.session_state["HPTCoolantBleed"] = ""
    st.session_state["LPTCoolantBleed"] = ""



# Now add a submit button to the form:

left, right = st.columns([1,1])
with left:
    is_submit = form.form_submit_button("Predict")
with right:
    is_clear = form.form_submit_button("Clear All", on_click=clear_form)
if is_submit:
    OpSet1 = 0 if OpSet1=='' else OpSet1
    OpSet2 = 0 if OpSet2 == '' else OpSet2
    LPCOutletTemp = 0 if LPCOutletTemp == '' else LPCOutletTemp
    HPCOutletTemp = 0 if HPCOutletTemp == '' else HPCOutletTemp
    LPTOutletTemp = 0 if LPTOutletTemp == '' else LPTOutletTemp
    TotalHPCOutletPressure = 0 if TotalHPCOutletPressure == '' else TotalHPCOutletPressure
    PhysicalFanSpeed = 0 if PhysicalFanSpeed == '' else PhysicalFanSpeed
    PhysicalCoreSpeed = 0 if PhysicalCoreSpeed == '' else PhysicalCoreSpeed
    StaticHPCOutletPressure = 0 if StaticHPCOutletPressure == '' else StaticHPCOutletPressure
    FuelFlowRatio = 0 if FuelFlowRatio == '' else FuelFlowRatio
    CorctFanSpeed = 0 if CorctFanSpeed == '' else CorctFanSpeed
    CorctCoreSpeed = 0 if CorctCoreSpeed == '' else CorctCoreSpeed
    BPR = 0 if BPR == '' else BPR
    HPTCoolantBleed = 0 if HPTCoolantBleed=='' else HPTCoolantBleed
    LPTCoolantBleed = 0 if LPTCoolantBleed == '' else LPTCoolantBleed



    names = ['OpSet1', 'OpSet2', 'LPCOutletTemp', 'HPCOutletTemp', 'LPTOutletTemp',
                         'TotalHPCOutletPressure', 'PhysicalFanSpeed', 'PhysicalCoreSpeed', 'StaticHPCOutletPressure',
                         'FuelFlowRatio', 'CorctFanSpeed', 'CorctCoreSpeed', 'BPR', 'HPTCoolantBleed',
                         'LPTCoolantBleed']
    # list_of_form_values = [[float(OpSet1)],[float(OpSet2)],[float(LPCOutletTemp)],[float(HPCOutletTemp)],[float(LPTOutletTemp)],[float(TotalHPCOutletPressure)],[float(PhysicalFanSpeed)],[float(PhysicalCoreSpeed)],
    #                        [float(StaticHPCOutletPressure)],[float(FuelFlowRatio)],[float(CorctFanSpeed)],[float(CorctCoreSpeed)],[float(BPR)],[float(HPTCoolantBleed)],
    #                        [float(LPTCoolantBleed)]]
    list_of_form_values = [float(OpSet1), float(OpSet2), float(LPCOutletTemp), float(HPCOutletTemp),
                           float(LPTOutletTemp), float(TotalHPCOutletPressure), float(PhysicalFanSpeed),
                           float(PhysicalCoreSpeed),
                           float(StaticHPCOutletPressure), float(FuelFlowRatio), float(CorctFanSpeed),
                           float(CorctCoreSpeed), float(BPR), float(HPTCoolantBleed),
                           float(LPTCoolantBleed)]
    # arr_form=st.write(np.transpose(np.array(list_of_form_values)))
    df_form = pd.DataFrame(data=[list_of_form_values], columns=names)
    # st.write(df_form)
    scaled_data_form = scaler.transform(df_form)
    # st.write(scaled_data_form)
    y_pred_temp = model.predict(scaled_data_form)
    y_pred_form = (y_pred_temp >= 0.5).astype('int')
    # st.write(y_pred_form[0])
    if y_pred_form[0]:
        # st.write("The remaining cycle of the engine could be less than 30 cycles.")
        st.warning('The remaining cycle of the engine could be less than 30 cycles.', icon="⚠️")
    else:
        # st.write("Engine is in good condition and can go more than 30 cycles")
        st.success('Engine is in good condition and can go more than 30 cycles!', icon="✅")




st.markdown("<h2 id='seperator' style='width: 100%; text-align: center; border-bottom: 1px dashed; line-height: 0.1em; margin: 10px 0 20px;'>"
            "<span style='background:rgb(14, 17, 23); padding:0 10px; color: white '>OR</span></h2>", unsafe_allow_html=True)


df_template = pd.DataFrame(columns=['EngineNo', 'Cycle','OpSet1', 'OpSet2', 'OpSet3','FanInletTemp', 'LPCOutletTemp', 'HPCOutletTemp', 'LPTOutletTemp', 'FanInletPressure',
              'ByPassDuctPressure', 'TotalHPCOutletPressure', 'PhysicalFanSpeed', 'PhysicalCoreSpeed',
              'EnginePressureRatio', 'StaticHPCOutletPressure', 'FuelFlowRatio', 'CorctFanSpeed', 'CorctCoreSpeed', 'BPR',
              'BurnerFuelRatio', 'BleedEnthalpy', 'DemandFanSpeed', 'DemandCorctFanSpeed', 'HPTCoolantBleed', 'LPTCoolantBleed'])
df_temp_csv=df_template.to_csv()
st.download_button(
        "Download Template",
        df_temp_csv,
        "template.csv",
        "text/csv",
        key='download-template'
    )

st.markdown("<h1 style='text-align: center; color: grey;'>Upload File</h1>", unsafe_allow_html=True)
csv=st.file_uploader("")
if csv is not None:
    df = pd.read_csv(csv)
    # st.write(df)

    df_filtered = df.filter(['EngineNo', 'Cycle', 'OpSet1', 'OpSet2', 'LPCOutletTemp', 'HPCOutletTemp', 'LPTOutletTemp',
                             'TotalHPCOutletPressure', 'PhysicalFanSpeed', 'PhysicalCoreSpeed',
                             'StaticHPCOutletPressure',
                             'FuelFlowRatio', 'CorctFanSpeed', 'CorctCoreSpeed', 'BPR', 'HPTCoolantBleed',
                             'LPTCoolantBleed'], axis=1)

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

    # plot_importance(model)

