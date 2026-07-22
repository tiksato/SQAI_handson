import numpy as np
import h5py
import os

def read_complex_column_stream(file_txt, usecols):
    with open(file_txt, "r") as f:
        for line in f:
            line = line.strip()
            if line == "" or line.startswith("#"):
                continue

            vals = line.split()
            yield float(vals[usecols[0]]) + 1j * float(vals[usecols[1]])

def write_h1_or_d1(file_txt, h5, dataset_name, usecols, nfun_full, chunk_size):
    if dataset_name in h5:
        print(f"{dataset_name} already exists.")
        return

    dset = h5.create_dataset(
        dataset_name,
        shape=(nfun_full, nfun_full),
        dtype=np.complex128,
        chunks=(1, min(256, nfun_full)),
        compression="gzip",
    )

    buffer = []
    row = 0
    total_size = nfun_full * nfun_full

    for z in read_complex_column_stream(file_txt, usecols):
        buffer.append(z)

        if len(buffer) == nfun_full:
            dset[row, :] = np.asarray(buffer, dtype=np.complex128)
            row += 1
            buffer = []

    if buffer:
        raise ValueError(
            f"{dataset_name}: incomplete final row. "
            f"Got {len(buffer)} values, expected {nfun_full}."
        )

    if row != nfun_full:
        raise ValueError(
            f"{dataset_name}: expected {nfun_full} rows, but read {row}."
        )

    print(f"{dataset_name} generated: shape = {dset.shape}")
    
def write_h2(file_txt, h5, dataset_name, usecols, nfun_full, chunk_size):
    if dataset_name in h5:
        print(f"{dataset_name} already exists.")
        return

    dset = h5.create_dataset(
        dataset_name,
        shape=(nfun_full, nfun_full, nfun_full, nfun_full),
        dtype=np.complex128,
        chunks=(1, 1, min(256, nfun_full), min(256, nfun_full)),
        compression="gzip",
    )

    buffer = []
    p = q = 0

    for z in read_complex_column_stream(file_txt, usecols):
        buffer.append(z)

        if len(buffer) == nfun_full * nfun_full:
            block = np.asarray(buffer, dtype=np.complex128).reshape(
                nfun_full, nfun_full
            )

            dset[p, q, :, :] = block

            q += 1
            if q == nfun_full:
                q = 0
                p += 1

            buffer = []

    if buffer:
        raise ValueError(
            f"{dataset_name}: incomplete final block. "
            f"Got {len(buffer)} values, expected {nfun_full * nfun_full}."
        )

    if p != nfun_full or q != 0:
        raise ValueError(
            f"{dataset_name}: incomplete data. Ended at p={p}, q={q}."
        )

    print(f"{dataset_name} generated: shape = {dset.shape}")

def write_flat_chunk_to_h2(dset, arr, offset, nfun_full):
    """
    Flat order assumed:
        (((p * nfun_full + q) * nfun_full + r) * nfun_full + s)

    i.e. h2[p, q, r, s].
    """

    for k, z in enumerate(arr):
        ind = offset + k

        s = ind % nfun_full
        ind //= nfun_full

        r = ind % nfun_full
        ind //= nfun_full

        q = ind % nfun_full
        ind //= nfun_full

        p = ind

        dset[p, q, r, s] = z

def active_orbital_indices(nang, nmax, nmax_full):
    return np.array(
        [l * nmax_full + n for l in range(nang) for n in range(nmax)],
        dtype=np.int64,
    )

def extract_active_h1(h5, nang, nmax):
    nmax_full = int(h5.attrs["nmax_full"])
    idx = active_orbital_indices(nang, nmax, nmax_full)

    h1 = h5["h1_full"][idx, :]
    h1 = h1[:, idx]

    return h1

def write_to_h5(file_int1e,
                file_int2e,
                file_dipole,
                file_h5,
                nang_full,
                nmax_full,
                usecols_h1 = (0,1), 
                usecols_h2 = (0,1), 
                usecols_d1 = (2,3),
                chunk_size = 100000,):

    nfun_full = nang_full * nmax_full
    
    with h5py.File(file_h5, "a") as h5:
        h5.attrs["nang_full"] = nang_full
        h5.attrs["nmax_full"] = nmax_full
        h5.attrs["nfun_full"] = nfun_full

        write_h1_or_d1(
            file_int1e,
            h5,
            "h1_full",
            usecols=usecols_h1,
            nfun_full=nfun_full,
            chunk_size=chunk_size,
        )

        write_h2(
            file_int2e,
            h5,
            "h2_full",
            usecols=usecols_h2,
            nfun_full=nfun_full,
            chunk_size=chunk_size,
        )

        write_h1_or_d1(
            file_dipole,
            h5,
            "d1_full",
            usecols=usecols_d1,
            nfun_full=nfun_full,
            chunk_size=chunk_size,
        )

def extract_active_d1(h5, nang, nmax):
    nmax_full = int(h5.attrs["nmax_full"])
    idx = active_orbital_indices(nang, nmax, nmax_full)

    d1 = h5["d1_full"][idx, :]
    d1 = d1[:, idx]

    return d1

def extract_active_h2(h5, nang, nmax):
    nmax_full = int(h5.attrs["nmax_full"])
    idx = active_orbital_indices(nang, nmax, nmax_full)

    h2_full = h5["h2_full"]

    # Fancy indexing from HDF5 only for the first axis
    h2 = h2_full[idx, :, :, :]

    h2 = h2[:, idx, :, :]
    h2 = h2[:, :, idx, :]
    h2 = h2[:, :, :, idx]

    # chemists' notation -> physicists' notation
    h2 = h2.transpose(0, 2, 1, 3)

    return h2

def load_integrals_atomic_nang_nmax(
        data_dir,
        nang, nmax,
        nang_full, nmax_full, 
        name_h5 = "integrals.h5",
        name_int1e = "int1e.dat", 
        name_int2e = "int2e.dat",
        name_dipole = "dipole.dat"):
    
    nfun = nang*nmax

    file_npz = f"{data_dir}/integrals_nang{nang}_nmax{nmax}.npz"
    file_h5 = f"{data_dir}/{name_h5}"
    file_int1e = f"{data_dir}/{name_int1e}"
    file_int2e = f"{data_dir}/{name_int2e}"
    file_dipole = f"{data_dir}/{name_dipole}"
    
    found = False
    
    if os.path.isfile(file_npz):
        with np.load(file_npz) as data:
            h1 = data["h1"]
            h2 = data["h2"]
            d1 = data["d1"]
        print(f"Integrals loaded from {file_npz}.")
        return h1, h2, d1
    
    for _nang in range(nang, nang_full+1):
        for _nmax in range(nmax, nmax_full+1):
            _file_npz = f"{data_dir}/integrals_nang{_nang}_nmax{_nmax}.npz"
            if os.path.isfile(_file_npz):
                nang_load = _nang
                nmax_load = _nmax
                with np.load(_file_npz) as data:
                    h1_loaded = data["h1"]
                    h2_loaded = data["h2"]
                    d1_loaded = data["d1"]
                    print("nang_load =", data["nang"])
                    print("nmax_load =", data["nmax"])
                print(f"Integrals loaded from {_file_npz}.")

                    
                tuple_h1 = (nang_load,nmax_load,)*2
                tuple_h2 = (nang_load,nmax_load,)*4
                slice_h1 = (slice(0,nang),slice(0,nmax))*2
                slice_h2 = (slice(0,nang),slice(0,nmax))*4
                h1 = h1_loaded.reshape(tuple_h1)[slice_h1].reshape((nfun,)*2)
                h2 = h2_loaded.reshape(tuple_h2)[slice_h2].reshape((nfun,)*4)
                d1 = d1_loaded.reshape(tuple_h1)[slice_h1].reshape((nfun,)*2)
                
                np.savez(file_npz, h1=h1, h2=h2, d1=d1, nang=nang, nmax=nmax)
                print(f"Integrals saved in {file_npz}.")
                return h1, h2, d1

    if os.path.isfile(file_h5):
        
        with h5py.File(file_h5, "r") as h5:
            h1 = extract_active_h1(h5, nang=nang, nmax=nmax)
            h2 = extract_active_h2(h5, nang=nang, nmax=nmax)
            d1 = extract_active_d1(h5, nang=nang, nmax=nmax)
        print(f"Integrals loaded from {file_h5}.")
            
        np.savez(file_npz, h1=h1, h2=h2, d1=d1, nang=nang, nmax=nmax)
        print(f"Integrals saved in {file_npz}.")
        return h1, h2, d1

        
    if os.path.isfile(file_int1e) and \
       os.path.isfile(file_int2e) and \
       os.path.isfile(file_dipole):
        
        write_to_h5(file_int1e, file_int2e, file_dipole, file_h5,
                    nang_full, nmax_full)
        print(f"Integrals read from dat files.")
        print(f"Integrals saved in {file_h5}.")
        
        with h5py.File(file_h5, "r") as h5:
            h1 = extract_active_h1(h5, nang=nang, nmax=nmax)
            h2 = extract_active_h2(h5, nang=nang, nmax=nmax)
            d1 = extract_active_d1(h5, nang=nang, nmax=nmax)
        print(f"Integrals loaded from {file_h5}.")
        
        np.savez(file_npz, h1=h1, h2=h2, d1=d1, nang=nang, nmax=nmax)
        print(f"Integrals saved in {file_npz}.")

        return h1, h2, d1

    raise FileNotFoundError(f"No integrals information found in {data_dir}")
