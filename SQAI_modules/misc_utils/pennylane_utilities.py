
# from qiskit import transpile
# from qiskit.quantum_info import Statevector
# from qiskit.quantum_info import SparsePauliOp
# from qiskit_ibm_runtime import SamplerV2, Session
# from qiskit.circuit import QuantumCircuit, QuantumRegister, Parameter
# from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
# import mthree

# def get_statevector(qc):
#    return(Statevector(qc).data)
#
# def weights_post_selection(weights, checker):
#    weights_ps = {}
#    wsum = 0.0
#    for c,w in weights.items():
#        if checker(c):
#            weights_ps[c] = w
#            wsum += w
#    weights_ps = {c:w/wsum for c,w in weights_ps.items()}
#    return(weights_ps)
#
# def calc_multi_Z_expectation(weights, target_bin):
#    #target_bin = int(target_bit,2)
#    Zexp = 0
#    for key, val in weights.items():
#        count_1 = bin(target_bin&int(key,2)).count('1')
#        phase = (-1)**count_1
#        Zexp += phase*val
#    return(Zexp)
#
# def calc_all_Z_expectations(weights, qubit_list):
#    Z_exp_dict = {}
#    for i in qubit_list:
#        target_bin = 2**i
#        Z_exp_dict[i] = calc_multi_Z_expectation(weights, target_bin)
#    return(Z_exp_dict)
#
# def calc_all_ZZ_expectations(weights, qubit_list):
#    ZZ_exp_dict = {}
#    for i,j in itertools.combinations(qubit_list,2):
#        target_bin = 2**i + 2**j
#        ZZ_exp_dict[(i,j)] = calc_multi_Z_expectation(weights, target_bin)
#    return(ZZ_exp_dict)
#
# class qiskit_sampler():
#    def __init__(self, pqc_list, backend, debug=False):
#        self.debug = debug
#        self.backend = backend
#        self.sampler = SamplerV2(mode=self.backend)
#        # for dynamical decoupling
#        self.samplerdd = SamplerV2(mode=self.backend)
#        self.samplerdd.options.dynamical_decoupling.enable = True
#        self.samplerdd.options.dynamical_decoupling.skip_reset_qubits = True
#        self.samplerdd.options.dynamical_decoupling.sequence_type = "XY4" # "XpXm"
#
#        self.gate_set = self.backend.configuration().basis_gates
#        self.pm = generate_preset_pass_manager(
#            backend=self.backend, optimization_level=3)
#
#        self.tpqc_list = []
#        for pqc in pqc_list:
#            pqc_measure = pqc.copy()
#            pqc_measure.measure_all()
#            self.tpqc_list.append(transpile(pqc_measure,
#                                   basis_gates=self.gate_set,
#                                   optimization_level=3))
#        self.isa_tpqc_list = self.pm.run(self.tpqc_list)
#
#        # for readout error mitigation
#        self.mit_list = []
#        self.mapping_list = []
#        for isa_tpqc in self.isa_tpqc_list:
#            self.mapping_list.append(mthree.utils.final_measurement_mapping(isa_tpqc))
#            self.mit_list.append(mthree.M3Mitigation(self.backend))
#
#        self.ps_checker = None
#        self.job = None
#        self.job_dd = None
#
#    def measure_all(self, params_list, shots):
#        #pubs = [(pqc,params) for pqc,params in zip(
#        #    self.isa_tpqc_list, params_list)]
#        pubs = []
#        for params in params_list:
#            for isa_tpqc in self.isa_tpqc_list:
#                pubs.append((isa_tpqc, params))
#        self.job = self.sampler.run(pubs=pubs, shots=shots)
#        self.results = self.job.result()
#        counts_list = [res.data.meas.get_counts()
#                       for res in self.results]
#        return(counts_list)
#
#    def measure_all_with_error_mitigation(self, params_list, shots):
#
#        if self.job is None and self.job_dd is None:
#            with Session(backend=self.backend) as session:
#                pubs = []
#                for params in params_list:
#                    for isa_tpqc in self.isa_tpqc_list:
#                        pubs.append((isa_tpqc, params))
#                self.job = self.sampler.run(pubs=pubs, shots=shots)
#                self.job_dd = self.samplerdd.run(pubs=pubs, shots=shots)
#
#                self.mapping0 = self.mapping_list[0]
#                self.mit0 = self.mit_list[0]
#                self.mit0.cals_from_system(self.mapping0, shots=shots, runtime_mode=session)
#
#        self.results = self.job.result()
#        self.results_dd = self.job_dd.result()
#        self.counts_list = [res.data.meas.get_counts()
#                            for res in self.results]
#        self.counts_dd_list = [res.data.meas.get_counts()
#                              for res in self.results_dd]
#
#        self.weights_list = [{key:c/shots for key,c in counts.items()}
#                             for counts in self.counts_list]
#        self.weights_m3_list = [self.mit0.apply_correction(counts, self.mapping0)
#                                for counts in self.counts_list]
#        self.weights_dd_list = [{key:c/shots for key,c in counts.items()}
#                                for counts in self.counts_dd_list]
#        self.weights_dd_m3_list = [self.mit0.apply_correction(counts, self.mapping0)
#                                   for counts in self.counts_dd_list]
#
#        if self.ps_checker is not None:
#            checker = self.ps_checker
#            self.weights_ps_list = [weights_post_selection(weights,checker)
#                                    for weights in self.weights_list]
#            self.weights_m3_ps_list = [weights_post_selection(weights,checker)
#                                       for weights in self.weights_m3_list]
#            self.weights_dd_ps_list = [weights_post_selection(weights,checker)
#                                       for weights in self.weights_dd_list]
#            self.weights_dd_m3_ps_list = [weights_post_selection(weights,checker)
#                                          for weights in self.weights_dd_m3_list]
#
#    def weights_statevector(self, pqc_list, params_list):
#        #loop_counter = 0
#
#        weights = []
#        for params in params_list:
#            for pqc in pqc_list:
#                qc = pqc.assign_parameters(params)
#                svec = Statevector(qc)
#                weights.append(svec.probabilities_dict())
#
#                #print('Loop counter', loop_counter)
#                #loop_counter += 1
#        return(weights)


class pair_ansatz:
    def __init__(
        self, n_occ, n_vir, order="occ_then_vir", ansatz_type="unitary_paired_double"
    ):

        self.n_occ = n_occ
        self.n_vir = n_vir
        self.order = order
        self.ansatz_type = ansatz_type

        self.n_qubits = n_occ + n_vir
        if self.ansatz_type == "gvb":
            self.n_parameters = n_occ
        else:
            self.n_parameters = n_occ * n_vir

    def get_circuits(self, add_measure=False):
        qc = QuantumCircuit(self.n_qubits)
        if self.order == "occ_then_vir":
            qc.x([j for j in range(self.n_occ)])
        else:
            qc.x([2 * j for j in range(self.n_occ)])

        i_param = 1
        if self.ansatz_type == "gvb":
            if self.order == "occ_then_vir":
                for i_occ in range(self.n_occ):
                    hole = i_occ
                    particle = self.n_occ + i_occ
                    qc.append(
                        self.get_an_excitation_gate(Parameter(f"p{i_param}")),
                        [hole, particle],
                    )
                    i_param += 1
            else:
                for i_occ in range(self.n_occ):
                    hole = 2 * i_occ
                    particle = 2 * i_occ + 1
                    qc.append(
                        self.get_an_excitation_gate(Parameter(f"p{i_param}")),
                        [hole, particle],
                    )
                    i_param += 1

        elif self.ansatz_type == "xgvb":
            for j_fun in range(self.n_occ + self.n_vir - 1):
                for i_fun in range(j_fun, self.n_occ + self.n_vir - 1, 2):
                    hole = i_fun
                    particle = i_fun + 1
                    qc.append(
                        self.get_an_excitation_gate(Parameter(f"p{i_param}")),
                        [hole, particle],
                    )
                    i_param += 1

        elif self.n_occ <= self.n_vir:
            for i_occ in range(self.n_occ):
                for a_vir in range(self.n_vir):
                    hole = self.n_occ - i_occ + a_vir - 1
                    particle = hole + 1
                    qc.append(
                        self.get_an_excitation_gate(Parameter(f"p{i_param}")),
                        [hole, particle],
                    )
                    i_param += 1
        else:
            for a_vir in range(self.n_vir):
                for i_occ in range(self.n_occ):
                    hole = self.n_occ - i_occ + a_vir - 1
                    particle = hole + 1
                    qc.append(
                        self.get_an_excitation_gate(Parameter(f"p{i_param}")),
                        [hole, particle],
                    )
                    i_param += 1
        return qc

    def get_an_excitation_gate(self, parameter):
        sub_q = QuantumRegister(2)
        sub_circ = QuantumCircuit(2, name="U")
        sub_circ.cx(1, 0)
        sub_circ.cry(2 * parameter, 0, 1)
        sub_circ.cx(1, 0)
        # Convert to a gate and stick it into
        # an arbitrary place in the bigger circuit
        sub_inst = sub_circ.to_instruction()
        return sub_inst


#    def get_doci_statevector(self, qc):
#        Cfock = get_statevector(qc)
#        CIC = []
#        n_orb = self.n_qubits
#        n_pair = self.n_occ
#        for pos1 in itertools.combinations(range(n_orb), n_pair):
#            c = ''.join(['1' if i in pos1 else '0'
#                         for i in range(n_orb)])
#            CIC.append(Cfock[int(c[::-1],2)].real)
#        return(np.asarray(CIC))
