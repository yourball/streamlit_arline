import streamlit as st
import streamlit.components.v1 as components

import datetime
from time import sleep, time
from stqdm import stqdm
import pandas as pd

from arline_quantum.gate_chain.gate_chain import GateChain
from arline_ml.compiler.strategies.strategy import Strategy
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

#
from arline_ml.compiler.strategies.strategy import Strategy as MlStrategy
import arline_benchmarks.pipeline.pipeline as benchmark_pipeline
benchmark_pipeline.Strategy = MlStrategy

# from arline_benchmarks.config_parser.pipeline_config_parser import PipelineConfigParser
# from arline_benchmarks.engines.pipeline_engine import PipelineEngine
# from arline_benchmarks.reports.latex_report import LatexReport
from PIL import Image

im = Image.open("arline.png")
st.set_page_config(page_title='ArlineQ', page_icon=im, layout="wide",)

# st.title("Arline Benchmarks")
st.markdown("""<p align="center"><h1 align="center">Arline Benchmarks</h1></p>""", unsafe_allow_html=True)



st.sidebar.markdown("## Quantum compilation frameworks")


add_qiskit = st.sidebar.checkbox("Qiskit", value=True)
add_tket = st.sidebar.checkbox("Tket", value=True)
add_cirq = st.sidebar.checkbox("Cirq", value=True)
add_voqc = st.sidebar.checkbox("VOQC", value=True)
add_pyzx = st.sidebar.checkbox("PyZX", value=True)



compilers_list = [{"Qiskit": add_qiskit},
                  {"Tket": add_tket},
                  {"Cirq": add_cirq},
                  {"VOQC": add_voqc},
                  {"PyZX": add_pyzx}]
num_compilers = sum([add_qiskit, add_tket, add_cirq, add_voqc, add_pyzx])



if num_compilers == 0:
    st.sidebar.error("At least one compiler should be selected")


# compiler_choices = ["Cirq", "Qiskit", "Tket", "VOQC", "PyZX"]
# compilers_list = []
# compiler = st.selectbox("Select compilers of interest",
#                              options=compiler_choices)
# compilers_list.append(compiler)
# add_compiler = st.button('Add another compiler')

# custom_pipeline = st.checkbox("Enable advanced mode: build custom compilation pipeline")
# strategy_list = ["arline_rebase",
#                  "pytket_mapping", "tket_default",
#                  "pytket_pauli_simp", "tket_peephole",
#                  "cx_directed", "tket_remove_redundances",
#                  "cirq_mapping", "cirq_eject_z", "cirq_eject_phased_paulis",
#                  "cirq_optimize_for_xmon", "pyzx_full_reduce", "pyzx_full_optimize",
#                  "qiskit_unroll",
#                  "qiskit_default", "voqc_default",
#                 ]
#
# if custom_pipeline:
#     pipeline_layers = st.multiselect('Choose a custom sequence of optimization subroutines',
#                     options=strategy_list)
#     st.write("My optimization pipeline", pipeline_layers)

st.markdown("#### ** Choose specifications for the input circuits **")

circ_options = ['from QASM', 'random Clifford+T', 'random Cnot+SU(2)']
circ_type = st.radio("Choose ", options=circ_options,)

random_target = circ_type in ['random Cnot+U3', 'random Clifford+T']
if random_target:
    col1, col2, col3 = st.columns(3)
    with col1:
        num_qubits_circ = st.number_input("Number of qubits in input circuit", min_value=1, max_value=20, step=1, value=10)
    with col2:
        num_gates = st.number_input("Number of gates in the input circuit", min_value=1, max_value=300, step=1, value=50)
    with col3:
        density_cx = st.slider("Density of Cnot gates", min_value=0., max_value=1., step=.1, value=0.5)
else:
    uploaded_file = st.file_uploader("Upload your OpenQASM file", type=['.qasm'])

st.sidebar.markdown("#### ** Quantum hardware **")
hardware_options = ["IBM All2All", "IBM Rueschlikon 16Q", "IBM Falcon 27Q",
                    "Google Sycamore 53Q", "Rigetti Agave 8Q", "Rigetti Aspen 16Q", "IonQ All2All"]
# col1, col2 = st.columns(2)
hardw_name = st.sidebar.selectbox("Choose target quantum hardware for compilation", options=hardware_options)
all2all_hardware = "All2All" in hardw_name



hardw_by_name_dict = {"IBM All2All": "IbmAll2All",
                "IBM Rueschlikon 16Q": "IbmRueschlikon",
                "IBM Falcon 27Q": "IbmFalcon",
                "Google Sycamore 53Q": "GoogleSycamore",
                "Rigetti Agave 8Q": "RigettiAgave",
                "Rigetti Aspen 16Q": "RigettiAspen",
                "IonQ All2All": "IonqAll2All",}



######################################
######################################
######################################
if all2all_hardware:
    num_qubits_hardw = st.sidebar.number_input(f"Please specify number of qubits in the target hardware",
                                    min_value=1, max_value=20, step=1, value=10)
else:
    st.sidebar.write(f"Selected backend: {hardw_name}")

if all2all_hardware:
    hw_cfg = {
            'class': hardw_by_name_dict[hardw_name],
            'args': {'num_qubits': num_qubits_hardw, }
          }
else:
    hw_cfg = {
            'class': hardw_by_name_dict[hardw_name],
            'args': {}
          }

click = st.sidebar.button('Run benchmark')
# Add Arline logo
logo = Image.open('logo.png')
st.sidebar.image(logo)


target_hw = hardware_by_name(hw_cfg)
st.sidebar.write('Output Gate Set:', target_hw.gate_set)

cirq_compression = {
    'id': 'cirq_compression',
    'strategy': 'cirq_mapping_compression',
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

qiskit_transpile = {
    'id': 'qiskit_transpile',
    'strategy': 'qiskit_transpile',
    'args': {
      'hardware': hw_cfg,
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
target_analysis = {
    'id': 'target_analysis',
    'strategy': 'target_analysis',
    'args': {
    },
}
run_analyser = True
QiskitPl = Pipeline(pipeline_id="Qiskit",
                    stages=[target_analysis, qiskit_transpile, arline_rebase], run_analyser=run_analyser)
PytketPl = Pipeline(pipeline_id="Pytket",
                    stages=[target_analysis, pytket_compression, arline_rebase], run_analyser=run_analyser)
CirqPl = Pipeline(pipeline_id="Cirq",
                  stages=[target_analysis, cirq_compression, arline_rebase], run_analyser=run_analyser)
PyZXPl = Pipeline(pipeline_id="PyZX",
                  stages=[target_analysis, pyzx_full_reduce, arline_rebase], run_analyser=run_analyser)
VoqcPl = Pipeline(pipeline_id="Voqc",
                  stages=[target_analysis, voqc, arline_rebase], run_analyser=run_analyser)

pipelines_list = []
if add_qiskit:
    pipelines_list.append(QiskitPl)
if add_tket:
    pipelines_list.append(PytketPl)
if add_cirq:
    pipelines_list.append(CirqPl)
if add_pyzx:
    pipelines_list.append(PyZXPl)
if add_voqc:
    pipelines_list.append(VoqcPl)


def get_random_targ_cfg(num_qubits, num_gates, circ_type):
    if circ_type == "random Clifford+T":
        hw_cfg = {
                  'hardware':  {
                      'class': 'CliffordTAll2All',
                      'args': {
                        'num_qubits': num_qubits_circ,
                      },
                    },
                  }
    if circ_type == "random Cnot+U3":
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

    for i, pl in enumerate(stqdm(pipelines_list)):
        new_chain = pl.run(target)

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
        st.write(pl.analyser_report_history)

    # df_1q = df_1q.set_index('Compiler')
    # df_2q = df_2q.set_index('Compiler')
    # df_time = df_time.set_index('Compiler')
    # df_check = df_check.set_index('Compiler')

    return df_1q, df_2q, df_time, df_check

proceed = True
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
        df_1q, df_2q, df_time, df_check = run_experiment(target)
        st.write('Benchmarking is finished. Please check the results below.')
        st.markdown("#### ** Results **")
        st.table(df_2q)
        expander_1q = st.expander("1Q gates stats")
        with expander_1q:
            st.table(df_1q)
        expander_time = st.expander("Runtime stats")
        with expander_time:
            st.table(df_time)
        expander_checks = st.expander("Validity checks")
        with expander_checks:
            st.table(df_check)
