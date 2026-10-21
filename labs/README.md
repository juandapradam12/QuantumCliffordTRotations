# PennyLane checks

These six scripts are the fixed-angle checks from the notebook folder. The package enumerates the same gates for every angle and every T-count. Each script pins one angle, or one named gate, and prints a state, a fidelity, or a Bloch vector.

They need PennyLane. From the repository root:

```bash
pip install -e ".[labs]"
python labs/Lab-U3.py
python labs/Lab-puertaT.py
python labs/Lab-Clifford+T.py
python labs/Lab-U3-HTScomp.py
python labs/Lab-norma.py
python labs/Lab-U3-HTScomp-Bloch.py
```

| Script | What it checks |
| --- | --- |
| `Lab-U3.py` | $U_3(\pi/2,\pi/2,\pi)$ on the ground state |
| `Lab-puertaT.py` | $R_x(\pi/4)$ against $HTH$, as a state fidelity |
| `Lab-Clifford+T.py` | $R_x(\pi/8)$ against the word $H\,T\,S^{\dagger}\,T\,H$ |
| `Lab-U3-HTScomp.py` | Bloch vectors of $U_3$, $H$, $S$, $S^{\dagger}$, $T$ and $T^{\dagger}$ on an equatorial state |
| `Lab-U3-HTScomp-Bloch.py` | The same comparison drawn on the Bloch sphere |
| `Lab-norma.py` | Normalization of a two-qubit state with amplitudes $(1,0,1,1)$ |

`Lab-U3-HTScomp-Bloch.py` opens a Matplotlib window. The other five scripts only print.
