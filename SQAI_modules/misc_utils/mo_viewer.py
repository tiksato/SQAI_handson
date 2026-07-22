from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import py3Dmol
from pyscf.tools import cubegen


class MolecularOrbitalViewer:
    """
    PySCFの分子軌道をJupyter上で表示するクラス。

    Parameters
    ----------
    mol
        PySCF Moleオブジェクト。
    mo_coeff
        RHF/RKSの場合:
            shape = (nao, nmo)

        UHF/UKSの場合:
            (mo_coeff_alpha, mo_coeff_beta)
            または shape = (2, nao, nmo)
    mo_energy
        MOエネルギー。省略可能。
    mo_occ
        MO占有数。省略可能。
    orbital_indices
        __init__でcubeデータを作成するMO番号。
        NoneならすべてのMOを作成する。
    spin
        UHF/UKSの場合に "alpha" または "beta"。
        省略時は "alpha"。
    isovalue
        MO等値面の絶対値。
    resolution
        cubeグリッドの間隔。単位はBohr。
    margin
        分子の周囲に確保するcube領域。単位はBohr。

    Notes
    -----
    MO番号はPythonと同じ0始まり。
    """

    def __init__(
        self,
        mol,
        mo_coeff,
        *,
        mo_energy=None,
        mo_occ=None,
        orbital_indices: Iterable[int] | None = None,
        spin: str | None = None,
        isovalue: float = 0.03,
        resolution: float = 0.4,
        margin: float = 4.0,
        positive_color: str = "blue",
        negative_color: str = "red",
        opacity: float = 0.75,
    ):
        self.mol = mol
        self.isovalue = abs(float(isovalue))
        self.resolution = float(resolution)
        self.margin = float(margin)
        self.positive_color = positive_color
        self.negative_color = negative_color
        self.opacity = float(opacity)

        (
            self.mo_coeff,
            self.mo_energy,
            self.mo_occ,
            self.spin,
        ) = self._select_orbitals(
            mo_coeff=mo_coeff,
            mo_energy=mo_energy,
            mo_occ=mo_occ,
            spin=spin,
        )

        if self.mo_coeff.ndim != 2:
            raise ValueError(
                "mo_coeff must have shape (nao, nmo), "
                f"but got {self.mo_coeff.shape}."
            )

        nao, self.nmo = self.mo_coeff.shape

        expected_nao = mol.nao_nr()
        if nao != expected_nao:
            raise ValueError(
                f"AO dimension mismatch: mo_coeff has {nao} rows, "
                f"but mol.nao_nr() is {expected_nao}."
            )

        if np.iscomplexobj(self.mo_coeff):
            imag_norm = np.linalg.norm(self.mo_coeff.imag)
            if imag_norm > 1.0e-12:
                raise ValueError(
                    "Complex molecular orbitals are not supported by this "
                    "viewer. Pass the real or imaginary part explicitly."
                )
            self.mo_coeff = self.mo_coeff.real

        if orbital_indices is None:
            indices = list(range(self.nmo))
        else:
            indices = list(dict.fromkeys(int(i) for i in orbital_indices))

        for index in indices:
            self._validate_mo_index(index)

        self.orbital_indices = tuple(indices)

        # {MO index: cube file contents}
        self._cube_data: dict[int, str] = {}

        # __init__で指定された一連のMOをcubeデータへ変換する。
        self._prepare_cube_data()

    @staticmethod
    def _is_unrestricted_coeff(mo_coeff) -> bool:
        """mo_coeffがUHF/UKS形式か判定する。"""
        if isinstance(mo_coeff, (tuple, list)):
            return (
                len(mo_coeff) == 2
                and np.asarray(mo_coeff[0]).ndim == 2
                and np.asarray(mo_coeff[1]).ndim == 2
            )

        array = np.asarray(mo_coeff)
        return array.ndim == 3 and array.shape[0] == 2

    @staticmethod
    def _select_spin_component(data, spin_index: int):
        """UHF/UKS形式の量からalphaまたはbeta成分を選ぶ。"""
        if data is None:
            return None

        if isinstance(data, (tuple, list)) and len(data) == 2:
            return np.asarray(data[spin_index])

        array = np.asarray(data)

        if array.ndim >= 2 and array.shape[0] == 2:
            return np.asarray(array[spin_index])

        return array

    def _select_orbitals(
        self,
        *,
        mo_coeff,
        mo_energy,
        mo_occ,
        spin,
    ):
        """RHF/UHF形式を統一した内部表現に変換する。"""
        is_unrestricted = self._is_unrestricted_coeff(mo_coeff)

        if not is_unrestricted:
            if spin is not None:
                raise ValueError(
                    "spin was specified, but mo_coeff is not in "
                    "UHF/UKS form."
                )

            return (
                np.asarray(mo_coeff),
                None if mo_energy is None else np.asarray(mo_energy),
                None if mo_occ is None else np.asarray(mo_occ),
                None,
            )

        if spin is None:
            spin = "alpha"

        spin = spin.lower()

        spin_aliases = {
            "alpha": 0,
            "a": 0,
            "α": 0,
            "beta": 1,
            "b": 1,
            "β": 1,
        }

        if spin not in spin_aliases:
            raise ValueError(
                "For UHF/UKS orbitals, spin must be 'alpha' or 'beta'."
            )

        spin_index = spin_aliases[spin]
        canonical_spin = "alpha" if spin_index == 0 else "beta"

        coeff = self._select_spin_component(mo_coeff, spin_index)
        energy = self._select_spin_component(mo_energy, spin_index)
        occupation = self._select_spin_component(mo_occ, spin_index)

        return coeff, energy, occupation, canonical_spin

    def _validate_mo_index(self, orbital_index: int) -> None:
        if not 0 <= orbital_index < self.nmo:
            raise IndexError(
                f"MO index {orbital_index} is outside "
                f"the valid range 0 <= index < {self.nmo}."
            )

    def _prepare_cube_data(self) -> None:
        """指定された全MOをcube文字列としてメモリ上に保持する。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            for orbital_index in self.orbital_indices:
                cube_path = tmpdir / f"mo_{orbital_index}.cube"

                cubegen.orbital(
                    self.mol,
                    str(cube_path),
                    self.mo_coeff[:, orbital_index],
                    resolution=self.resolution,
                    margin=self.margin,
                )

                self._cube_data[orbital_index] = cube_path.read_text(
                    encoding="utf-8"
                )

    def _title(self, orbital_index: int) -> str:
        """各ビューに表示するMO情報を構築する。"""
        if self.spin is None:
            parts = [f"MO index = {orbital_index}"]
        else:
            spin_symbol = "alpha" if self.spin == "alpha" else "beta"
            parts = [f"{spin_symbol} MO index = {orbital_index}"]

        if self.mo_energy is not None:
            energy = float(self.mo_energy[orbital_index])
            parts.append(f"E = {energy:.6f} Eh")

        if self.mo_occ is not None:
            occupation = float(self.mo_occ[orbital_index])
            parts.append(f"occ = {occupation:g}")

        return "    ".join(parts)

    def _add_orbital(
        self,
        view,
        orbital_index: int,
        *,
        viewer: tuple[int, int] | None = None,
        isovalue: float | None = None,
        show_title: bool = True,
    ) -> None:
        """一つのpy3Dmol viewerへ軌道を追加する。"""
        self._validate_mo_index(orbital_index)

        if orbital_index not in self._cube_data:
            raise KeyError(
                f"MO {orbital_index} was not loaded in __init__. "
                f"Loaded MOs are {list(self.orbital_indices)}."
            )

        cube_data = self._cube_data[orbital_index]
        value = self.isovalue if isovalue is None else abs(float(isovalue))

        target = {} if viewer is None else {"viewer": viewer}

        # cube内の原子座標を分子模型として追加
        view.addModel(cube_data, "cube", **target)

        view.setStyle(
            {},
            {
                "stick": {"radius": 0.12},
                "sphere": {"scale": 0.25},
            },
            **target,
        )

        # MOの正の位相
        view.addVolumetricData(
            cube_data,
            "cube",
            {
                "isoval": value,
                "color": self.positive_color,
                "opacity": self.opacity,
            },
            **target,
        )

        # MOの負の位相
        view.addVolumetricData(
            cube_data,
            "cube",
            {
                "isoval": -value,
                "color": self.negative_color,
                "opacity": self.opacity,
            },
            **target,
        )

        if show_title:
            # 画面左上に固定されたラベル
            view.addLabel(
                self._title(orbital_index),
                {
                    "position": {"x": 8, "y": 8, "z": 0},
                    "useScreen": True,
                    "inFront": True,
                    "fontSize": 13,
                    "fontColor": "black",
                    "backgroundColor": "white",
                    "backgroundOpacity": 0.75,
                    "borderThickness": 0,
                },
                **target,
            )

        view.zoomTo(**target)

    def show(
        self,
        orbital_index: int,
        *,
        isovalue: float | None = None,
        width: int = 700,
        height: int = 500,
        show_title: bool = True,
    ):
        """
        一つの分子軌道を表示する。

        Example
        -------
        viewer.show(homo)
        """
        view = py3Dmol.view(width=width, height=height)

        self._add_orbital(
            view,
            orbital_index,
            isovalue=isovalue,
            show_title=show_title,
        )

        return view.show()

    def show_list(
        self,
        orbital_indices: Sequence[int] | None = None,
        *,
        ncols: int,
        nrows: int,
        isovalue: float | None = None,
        cell_width: int = 320,
        cell_height: int = 280,
        linked: bool = True,
        show_title: bool = True,
    ):
        """
        複数の分子軌道を指定された行数・列数で表示する。

        Parameters
        ----------
        orbital_indices
            表示するMO番号。Noneなら__init__で読み込んだ全MO。
        ncols
            列数。
        nrows
            行数。
        linked
            Trueなら全ビューを同時に回転・拡大縮小する。

        Example
        -------
        viewer.show_list(
            [homo - 2, homo - 1, homo, lumo, lumo + 1],
            ncols=3,
            nrows=2,
        )
        """
        if ncols <= 0 or nrows <= 0:
            raise ValueError("ncols and nrows must both be positive.")

        if orbital_indices is None:
            indices = list(self.orbital_indices)
        else:
            indices = [int(i) for i in orbital_indices]

        capacity = ncols * nrows

        if len(indices) > capacity:
            raise ValueError(
                f"{len(indices)} orbitals were requested, but a "
                f"{nrows} x {ncols} grid has only {capacity} cells."
            )

        for orbital_index in indices:
            self._validate_mo_index(orbital_index)

            if orbital_index not in self._cube_data:
                raise KeyError(
                    f"MO {orbital_index} was not loaded in __init__."
                )

        view = py3Dmol.view(
            width=cell_width * ncols,
            height=cell_height * nrows,
            viewergrid=(nrows, ncols),
            linked=linked,
        )

        for position, orbital_index in enumerate(indices):
            row, col = divmod(position, ncols)

            self._add_orbital(
                view,
                orbital_index,
                viewer=(row, col),
                isovalue=isovalue,
                show_title=show_title,
            )

        return view.show()

    def __len__(self) -> int:
        """__init__で読み込んだMO数。"""
        return len(self.orbital_indices)

    def __repr__(self) -> str:
        spin_info = "" if self.spin is None else f", spin={self.spin!r}"

        return (
            f"{self.__class__.__name__}("
            f"nmo={self.nmo}, "
            f"loaded={len(self)}"
            f"{spin_info})"
        )

