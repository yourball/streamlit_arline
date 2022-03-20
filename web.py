import streamlit as st
import streamlit.components.v1 as components

import numpy as np
import re

import datetime
from time import sleep, time
# from stqdm import stqdm
import pandas as pd
from PIL import Image

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from arline_quantum.gate_chain.gate_chain import GateChain
from arline_benchmarks.pipeline.pipeline import Pipeline
from arline_quantum.qubit_connectivities.qubit_connectivity import QubitConnectivity
from arline_quantum.hardware import Hardware
from arline_quantum.gate_sets.gate_set import GateSet
from arline_quantum.gates.cnot import Cnot
from arline_quantum.gates.u3 import U3
from arline_benchmarks.targets.target import Target
from arline_benchmarks.strategies.qiskit_transpile import QiskitTranspile
from arline_quantum.hardware import hardware_by_name
from arline_benchmarks.engines.pipeline_engine import PipelineEngine
import arline_benchmarks.pipeline.pipeline as benchmark_pipeline

im = Image.open("arline.png")
st.set_page_config(page_title='ArlineQ', page_icon=im, layout="wide",)

st.markdown("""<p align="center"><h1 align="center">Arline Benchmarks</h1></p>""", unsafe_allow_html=True)

logo = Image.open('logo.png')
st.sidebar.image(logo)

st.sidebar.markdown("""<hr style="height:2px;color:white;"></hr>""", unsafe_allow_html=True)
st.sidebar.markdown("## Quantum compilation frameworks")

add_qiskit = st.sidebar.checkbox("Qiskit", value=True)
add_tket = st.sidebar.checkbox("Tket", value=True)
add_cirq = st.sidebar.checkbox("Cirq", value=True)
add_pyzx = st.sidebar.checkbox("PyZX", value=True)
# add_voqc = st.sidebar.checkbox("VOQC", value=True)


compilers_list = [{"Qiskit": add_qiskit},
                  {"Tket": add_tket},
                  {"Cirq": add_cirq},
                  {"PyZX": add_pyzx},
                  # {"VOQC": add_voqc},
                  ]

num_compilers = sum([add_qiskit, add_cirq, add_pyzx, add_tket,  # add_voqc,
                    ])
if num_compilers == 0:
    st.sidebar.error("At least one compiler should be selected")


st.sidebar.markdown("""<hr style="height:2px;color:white;"></hr>""", unsafe_allow_html=True)

st.sidebar.markdown("## Quantum hardware")
hardware_options = ["IBM All2All", "IBM Rueschlikon 16Q", "IBM Falcon 27Q",
                    "Google Sycamore 53Q", "Rigetti Aspen 16Q", "IonQ All2All"]
hardw_name = st.sidebar.selectbox("Choose target quantum hardware for compilation", options=hardware_options)
all2all_hardware = "All2All" in hardw_name

routing_only = st.sidebar.checkbox("Qubit routing only (ignore circuit compression passes)")

hardw_by_name_dict = {"IBM All2All": "IbmAll2All",
                "IBM Rueschlikon 16Q": "IbmRueschlikon",
                "IBM Falcon 27Q": "IbmFalcon",
                "Google Sycamore 53Q": "GoogleSycamore",
                "Rigetti Aspen 16Q": "RigettiAspen",
                "IonQ All2All": "IonqAll2All",}

if all2all_hardware:
    num_qubits_hardw = st.sidebar.number_input(f"Specify number of qubits in the target hardware",
                                    min_value=1, max_value=20, step=1, value=10)
    hw_cfg = {
            'class': hardw_by_name_dict[hardw_name],
            'args': {'num_qubits': num_qubits_hardw, }
          }
else:
    st.sidebar.write(f"Selected backend: {hardw_name}")
    hw_cfg = {
            'class': hardw_by_name_dict[hardw_name],
            'args': {}
          }

target_hw = hardware_by_name(hw_cfg)
st.sidebar.markdown(f"Native Gate Set: {target_hw.gate_set.get_gate_names()}")

num_qubits_hardw = target_hw.qubit_connectivity._num_qubits

st.markdown("""<h2 style="color:grey"> Choose specifications for the input circuits </h2>""", unsafe_allow_html=True)

circ_options = ['From QASM file', 'Random Clifford+T circuit', 'Random Cnot+SU(2) circuit',]
circ_type = st.radio("Input circuit type", options=circ_options,)

random_target = 'Random' in circ_type

if random_target:
    with st.expander("How random circuits are generated?"):
        st.write("""CNOT gates are sampled such that the locations of the control and target qubits are drawn from the uniform distribution.
        The total number of CNOTs is controlled by the CNOT density parameter.
        The type of discrete single-qubit gates and their placement is drown from uniform distribution.
        The continuiuous angles in single-qubit SU(2) gates are sampled from the Haar distribution.""")

if random_target:
    col1, col2 = st.columns(2)
    with col1:
        num_gates = st.number_input("Number of gates in the input circuit", min_value=1, max_value=300, step=1, value=50)
    with col2:
        density_cx = st.slider("Density of Cnot gates", min_value=0., max_value=1., step=.1, value=0.5)
    num_gates = np.int(num_gates)
    num_qubits_circ = num_qubits_hardw
else:
    uploaded_file = st.file_uploader("Upload your OpenQASM file", type=['.qasm'])

# m = st.markdown("""
# <style>
# div.stButton > button:first-child {
#   background-color: #4CAF50; /*#404040;*/
#   border: none;
#   color: white;
#   padding: 15px 32px;
#   text-align: center;
#   text-decoration: none;
#   display: inline-block;
#   font-size: 16px;
#   border-radius: 13px;
# }
# </style>""", unsafe_allow_html=True)

m = st.markdown("""
<style>
div.stButton > button:first-child {
    width: 10em;
    height: 4em;
    font-size: 18px;
    background-color: #4CAF50;
    color:white;
    font-size: 16px;
    border-radius: 13px;
}
div.stButton > button:hover {
    background-color: #0c9131;
    border-color:none;
    border-width: 0;
    color: white;
    outline: currentcolor none medium;
    }
div.stButton > button:focus {
    background-color: #4CAF50;
    color:white;
    border-width: 0;
    outline: none;
    outline: currentcolor none medium;
    }
</style>""", unsafe_allow_html=True)

click = st.button('Run benchmark')

target_analysis = {
    'id': 'target_analysis',
    'strategy': 'target_analysis',
    'args': {
    },
}
cirq_compression = {
    'id': 'cirq_compression',
    'strategy': 'cirq_mapping_compression',
    'args': {
      'hardware': hw_cfg,
    },
  }
cirq_mapping = {
    'id': 'cirq_mapping',
    'strategy': 'cirq_mapping',
    'args': {
      'hardware': hw_cfg,
    },
  }
voqc = {
    'id': 'voqc',
    'strategy': 'voqc',
    'args': {
      'hardware': hw_cfg,
    },
  }

pytket_compression = {
    'id': 'pytket_compression',
    'strategy': 'pytket_mapping_compression',
    'args': {
      'hardware': hw_cfg,
    },
  }

pytket_mapping = {
    'id': 'pytket_mapping',
    'strategy': 'pytket_mapping',
    'args': {
      'hardware': hw_cfg,
    },
  }

qiskit_compression = {
    'id': 'qiskit_compression',
    'strategy': 'qiskit_transpile',
    'args': {
      'hardware': hw_cfg,
    },
  }

qiskit_mapping = {
    'id': 'qiskit_mapping',
    'strategy': 'qiskit_transpile',
    'args': {
      'hardware': hw_cfg,
      'optimization_level': 0
    },
  }

pyzx_full_reduce = {
    'id': 'pyzx_full_reduce',
    'strategy': 'pyzx_full_reduce',
    'args': {
      'hardware': hw_cfg,
    },
  }

arline_rebase = {
    'id': 'arline_rebase',
    'strategy': 'arline_rebase',
    'args': {
      'hardware': hw_cfg,
    },
}

if routing_only:
    qiskit_stages = [target_analysis, qiskit_mapping, arline_rebase]
    pytket_stages = [target_analysis, pytket_mapping, arline_rebase]
    cirq_stages = [target_analysis, cirq_mapping, arline_rebase]
    pyzx_stages = [target_analysis, qiskit_mapping, arline_rebase]
else:
    qiskit_stages = [target_analysis, qiskit_compression, arline_rebase]
    pytket_stages = [target_analysis, pytket_compression, arline_rebase]
    cirq_stages = [target_analysis, cirq_compression, arline_rebase]
    pyzx_stages = [target_analysis, pyzx_full_reduce, arline_rebase]

QiskitPl = Pipeline(pipeline_id="Qiskit", stages=qiskit_stages, run_analyser=True)
PytketPl = Pipeline(pipeline_id="Pytket", stages=pytket_stages, run_analyser=True)
CirqPl = Pipeline(pipeline_id="Cirq", stages=cirq_stages, run_analyser=True)
PyZXPl = Pipeline(pipeline_id="PyZX", stages=pyzx_stages, run_analyser=True)
# VoqcPl = Pipeline(pipeline_id="Voqc",
#                   stages=[target_analysis, voqc, arline_rebase], run_analyser=run_analyser)

pipelines_list = []
if add_qiskit:
    pipelines_list.append(QiskitPl)
if add_tket:
    pipelines_list.append(PytketPl)
if add_cirq:
    pipelines_list.append(CirqPl)
if add_pyzx:
    pipelines_list.append(PyZXPl)
# if add_voqc:
#     pipelines_list.append(VoqcPl)


def get_random_targ_cfg(num_qubits, num_gates, circ_type):
    if circ_type == circ_options[1]:
        hw_cfg = {
                  'hardware':  {
                      'class': 'CliffordTAll2All',
                      'args': {
                        'num_qubits': num_qubits_circ,
                      },
                    },
                  }
    if circ_type == circ_options[2]:
        hw_cfg = {
                  "hardware": {
                       "gate_set": [
                          "U3",
                          "Cnot"
                       ],
                       "num_qubits": num_qubits_circ
                    },
                }
    cfg = {
      'task': 'circuit_transformation',
      'name': circ_type,
      'algo': 'random_chain',
      'number': 1,
      'seed': 1,
      'gate_distribution': {'Cnot': density_cx},
      'chain_length': num_gates,
    }
    cfg.update(hw_cfg)
    return cfg


def run_experiment(target):
    columns_1q = ['Compiler', '1Q gates count before', '1Q gates depth before',
               '1Q gates count after', '1Q gates depth after']
    columns_2q = ['Compiler', '2Q gates count before', '2Q gates depth before',
               '2Q gates count after', '2Q gates depth after']
    columns_time = ['Compiler', 'Execution time (seconds)']
    columns_check = ['Compiler', 'Connectivity Satisfied', 'Gateset Satisfied']
    df_1q = pd.DataFrame(columns=columns_1q)
    df_2q = pd.DataFrame(columns=columns_2q)
    df_time = pd.DataFrame(columns=columns_time)
    df_check = pd.DataFrame(columns=columns_check)
    df_full = pd.DataFrame()

    pl_history_list = []
    progress_bar = st.progress(0)
    for i, pl in enumerate(pipelines_list):
        new_chain = pl.run(target)
        progress_bar.progress(int(100*(i+1)/len(pipelines_list)))
        g_single_qubit_before = pl.analyser_report_history[0]["Single-Qubit Gate Count"]
        d_single_qubit_before = pl.analyser_report_history[0]["Single-Qubit Gate Depth"]
        g_single_qubit_after = pl.analyser_report_history[-1]["Single-Qubit Gate Count"]
        d_single_qubit_after = pl.analyser_report_history[-1]["Single-Qubit Gate Depth"]

        g_count_before = pl.analyser_report_history[0]["Two-Qubit Gate Count"]
        d_cnot_before = pl.analyser_report_history[0]["Two-Qubit Gate Depth"]
        g_count_after = pl.analyser_report_history[-1]["Two-Qubit Gate Count"]
        d_cnot_after = pl.analyser_report_history[-1]["Two-Qubit Gate Depth"]

        df_1q.loc[i] = [pl.id, g_single_qubit_before, d_single_qubit_before, g_single_qubit_after, d_single_qubit_after]
        df_2q.loc[i] = [pl.id, g_count_before, d_cnot_before, g_count_after, d_cnot_after]
        df_time.loc[i] = [pl.id, pl.analyser_report_history[-2]["Total Execution Time"]]
        df_check.loc[i] = [pl.id,
                           pl.analyser_report_history[-1]["Connectivity Satisfied"],
                           pl.analyser_report_history[-1]["Gate Set Satisfied"]]
        df_tmp = pd.DataFrame(pl.analyser_report_history)
        df_tmp['Compiler'] = pl.id
        pl_history_list.append(df_tmp)
    df_full = pd.concat(pl_history_list, axis=0)
    return df_1q, df_2q, df_time, df_check, df_full


def plot_gate_composition(data, compiler_name):
    data = data[data['Compiler'] == compiler_name]

    # data = data.reset_index().groupby(by="Stage").aggregate(np.sum)
    data = data.reset_index()
    data = data.sort_values("index", ascending=False)

    g_count_sum = data.filter(regex=("Count of .* Gates"), axis=1)
    g_count_sum = g_count_sum.loc[:, (data != 0).any(axis=0)]

    old_g_names = g_count_sum.columns
    g_names = [re.search("Count of (.*) Gates", g).group(1) for g in old_g_names]
    stage_names = ['Target', 'Mapping/Compression', 'Rebase']

    trace = go.Heatmap(
           x=g_names,
           y=np.flip(np.array(stage_names)),
           z=g_count_sum,
           type='heatmap',
           colorscale='Blues'
        )
    fig = go.Figure(data=[trace], layout=go.Layout(title=go.layout.Title(text=f"{compiler_name}")))
    st.plotly_chart(fig, use_container_width=True)

proceed = True
if random_target and num_qubits_hardw < num_qubits_circ:
    st.error("Number of qubits in the quantum circuit must be smaller then number of qubits in the quantum hardware.")
if click:
    if random_target:
        targ_cfg = get_random_targ_cfg(num_qubits_circ, num_gates, circ_type)
        target_generator = Target.from_config(config=targ_cfg)
        target = next(target_generator)[0]
    else:
        if uploaded_file is not None:
            qasm_data = str(uploaded_file.read(), "utf-8")
            target = GateChain.from_qasm_string(qasm_data)
        else:
            st.error("QASM file is not uploaded")
            proceed = False

    if proceed:
        df_1q, df_2q, df_time, df_check, df_full = run_experiment(target)
        if not all2all_hardware and add_pyzx:
            st.warning('PyZX does not contain qubit routing module. Using Qiskit Transpile for routing instead.')
        st.success('Benchmarking is finished. Please check the results below.')
        st.sidebar.markdown("""<hr style="height:4px;color:white;width:100%"></hr>""", unsafe_allow_html=True)
        st.markdown("""<h2 style="color:grey"> Results </h2>""", unsafe_allow_html=True)
        st.write("""
        """)

        with st.container():
            df_merge = pd.concat([df_1q, df_2q.drop(['Compiler'], axis=1)], axis=1)
            features = ['1Q gates count', '1Q gates depth', '2Q gates count', '2Q gates depth']
            for cc in features:
                df_merge[cc+' ratio'] = df_merge[cc+' before'].divide(df_merge[cc+' after'])

            df_merge = df_merge.set_index('Compiler')

            col1, col2 = st.columns(2)
            with col1:
                fig_radar = go.Figure(layout=go.Layout(title=go.layout.Title(text="Compression ratio (higher is better)")))
                theta = [cc+' ratio' for cc in features]
                for idx in range(len(df_merge)):
                    fig_radar.add_trace(go.Scatterpolar(r=df_merge.iloc[idx].values[-4:],
                                                        theta=theta,
                                                        fill='toself',
                                                        name=df_merge.iloc[idx].name
                                                        ))
                st.plotly_chart(fig_radar, use_container_width=True)

            with col2:
                fig_2q = go.Figure(data=[
                    go.Bar(name='2Q gates count', x=df_2q['Compiler'].values, y=df_2q['2Q gates count after']),
                    go.Bar(name='2Q gates depth', x=df_2q['Compiler'].values, y=df_2q['2Q gates depth after']),
                    ],
                    layout=go.Layout(title=go.layout.Title(text="2Q metrics after compression (lower is better)"))
                    )
                fig_2q.update_layout(barmode='group')
                st.plotly_chart(fig_2q, use_container_width=True)

            st.write("""We define Compression Ratio (CR) for the metrics of interest (gate count, gate depth) as:""")
            st.latex(r''' CR (\textrm{gate count}, g \in G) = \frac{\textrm{gate count}_{in}}{\textrm{gate count}_{out}} ''')
            st.latex(r''' CR (\textrm{gate depth}, g \in G) = \frac{\textrm{gate depth}_{in}}{\textrm{gate depth}_{out}} ''')

            st.table(df_2q)

            expander_1q = st.expander("1Q gates stats")
            with expander_1q:
                fig_1q = go.Figure(data=[
                    go.Bar(name='1Q gates count', x=df_1q['Compiler'].values, y=df_1q['1Q gates count after']),
                    go.Bar(name='1Q gates depth', x=df_1q['Compiler'].values, y=df_1q['1Q gates depth after']),
                    ],
                    layout=go.Layout(title=go.layout.Title(text="1Q metrics after compression (lower is better)"))
                    )
                fig_1q.update_layout(barmode='group')
                st.plotly_chart(fig_1q, use_container_width=True)
                st.table(df_1q)

            with st.expander("Gate composition (gate counts) for each compilation stage"):
                compiler_names = df_full['Compiler'].unique()
                cols = st.columns(len(compiler_names))
                for idx in range(len(cols)):
                    with cols[idx]:
                        plot_gate_composition(df_full, compiler_names[idx])

            expander_time = st.expander("Runtime stats")
            with expander_time:
                fig_time = px.bar(df_time, x='Compiler', y='Execution time (seconds)')
                st.plotly_chart(fig_time, use_container_width=True)
            expander_checks = st.expander("Validity checks")
            with expander_checks:
                st.table(df_check)
