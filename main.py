from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
import numpy as np
import matplotlib
#matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import random

# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 1: HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def encode_qubit(bit: int, basis: str) -> QuantumCircuit:
    """
    Alice encodes a classical bit into a qubit.

    Encoding rules (BB84):
      Rectilinear basis ('+'):
          bit=0  →  |0⟩  (no gate needed)
          bit=1  →  |1⟩  (apply Pauli-X)
      Diagonal basis ('x'):
          bit=0  →  |+⟩  (apply Hadamard)
          bit=1  →  |−⟩  (apply Pauli-X, then Hadamard)

    Args:
        bit   : 0 or 1
        basis : '+' (rectilinear) or 'x' (diagonal)

    Returns:
        QuantumCircuit (1 qubit, no measurement yet)
    """
    qc = QuantumCircuit(1, 1)
    if bit == 1:
        qc.x(0)          # Flip to |1⟩ if bit is 1
    if basis == 'x':
        qc.h(0)          # Apply Hadamard to enter diagonal basis
    return qc


def measure_qubit(qc: QuantumCircuit, basis: str) -> QuantumCircuit:
    """
    Bob measures the qubit in his chosen basis.

    Measurement rules:
      Rectilinear basis ('+'):  measure directly in computational basis
      Diagonal basis ('x'):     apply Hadamard first, then measure

    Args:
        qc    : QuantumCircuit with Alice's encoded qubit
        basis : '+' or 'x'

    Returns:
        QuantumCircuit with measurement appended
    """
    if basis == 'x':
        qc.h(0)          # Rotate back from diagonal basis before measuring
    qc.measure(0, 0)
    return qc


def run_bb84_protocol(n_bits: int = 20, eavesdrop: bool = False):
    """
    Full BB84 simulation.

    Steps:
      1. Alice generates random bits and random bases.
      2. Alice encodes each bit into a qubit.
      3. (Optional) Eve intercepts and re-encodes (eavesdropping simulation).
      4. Bob chooses random bases and measures.
      5. Alice and Bob publicly compare bases (sifting).
      6. Kept bits where bases matched form the raw key.
      7. Error rate is computed to detect eavesdropping.

    Args:
        n_bits    : Number of qubits to transmit
        eavesdrop : If True, simulate an eavesdropper (Eve)

    Returns:
        dict with all protocol data
    """
    simulator = AerSimulator()

    # ── Step 1: Alice's random bits and bases ────────────────────────────────
    alice_bits  = [random.randint(0, 1) for _ in range(n_bits)]
    alice_bases = [random.choice(['+', 'x']) for _ in range(n_bits)]

    # ── Step 2 (& 3): Encode → (Eve intercepts) → Bob measures ──────────────
    bob_bases   = [random.choice(['+', 'x']) for _ in range(n_bits)]
    bob_results = []
    eve_bases   = [random.choice(['+', 'x']) for _ in range(n_bits)] if eavesdrop else []

    for i in range(n_bits):
        # Alice encodes
        qc = encode_qubit(alice_bits[i], alice_bases[i])

        if eavesdrop:
            # Eve measures in her random basis (collapses the qubit)
            eve_qc = qc.copy()
            if eve_bases[i] == 'x':
                eve_qc.h(0)
            eve_qc.measure(0, 0)
            job = simulator.run(eve_qc, shots=1)
            eve_result = int(list(job.result().get_counts().keys())[0])

            # Eve re-encodes what she measured and sends to Bob
            qc = encode_qubit(eve_result, eve_bases[i])

        # Bob measures in his chosen basis
        qc = measure_qubit(qc, bob_bases[i])
        job = simulator.run(qc, shots=1)
        result = int(list(job.result().get_counts().keys())[0])
        bob_results.append(result)

    # ── Step 4: Sifting — keep only bits where bases matched ─────────────────
    sifted_alice = []
    sifted_bob   = []
    matched_indices = []

    for i in range(n_bits):
        if alice_bases[i] == bob_bases[i]:
            sifted_alice.append(alice_bits[i])
            sifted_bob.append(bob_results[i])
            matched_indices.append(i)

    # ── Step 5: Error rate calculation ───────────────────────────────────────
    if len(sifted_alice) > 0:
        errors    = sum(a != b for a, b in zip(sifted_alice, sifted_bob))
        error_rate = errors / len(sifted_alice) * 100
    else:
        errors, error_rate = 0, 0.0

    # Final shared key (bits that matched perfectly, used after verification)
    shared_key = sifted_alice  # In a real system, Alice's bits are ground truth

    return {
        'n_bits'          : n_bits,
        'alice_bits'      : alice_bits,
        'alice_bases'     : alice_bases,
        'bob_bases'       : bob_bases,
        'bob_results'     : bob_results,
        'matched_indices' : matched_indices,
        'sifted_alice'    : sifted_alice,
        'sifted_bob'      : sifted_bob,
        'shared_key'      : shared_key,
        'errors'          : errors,
        'error_rate'      : error_rate,
        'eavesdrop'       : eavesdrop,
    }


# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 2: VISUALIZATION
# ─────────────────────────────────────────────────────────────────────────────

def visualize_results(res_no_eve: dict, res_with_eve: dict, save_path: str = 'bb84_results.png'):
    """
    Produce a comprehensive 6-panel figure summarising the BB84 simulation:
      Panel 1 : Example encoding circuits (Alice)
      Panel 2 : Basis comparison table (first 10 bits, no Eve)
      Panel 3 : Sifted key bits — no Eve
      Panel 4 : Sifted key bits — with Eve
      Panel 5 : Error rate comparison bar chart
      Panel 6 : Key length retention comparison
    """
    fig = plt.figure(figsize=(20, 14), facecolor='white')
    fig.suptitle(
        'AIM3231 Mini Project: BB84 Quantum Key Distribution Protocol\n'
        'Manipal University Jaipur — Qiskit Simulation',
        fontsize=16, fontweight='bold', y=0.98
    )

    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.55, wspace=0.38,
                           top=0.91, bottom=0.06, left=0.06, right=0.97)

    # ── COLOUR PALETTE ────────────────────────────────────────────────────────
    C_GREEN  = '#27ae60'
    C_RED    = '#e74c3c'
    C_BLUE   = '#2980b9'
    C_ORANGE = '#e67e22'
    C_PURPLE = '#8e44ad'
    C_GREY   = '#95a5a6'
    C_LIGHT  = '#ecf0f1'

    # ═════════════════════════════════════════════════════════════════════════
    #  PANEL 1 – Four example encoding circuits
    # ═════════════════════════════════════════════════════════════════════════
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.axis('off')
    ax1.set_title('BB84 Encoding Circuits (Alice)', fontweight='bold',
                  fontsize=11, pad=8)

    circuits_info = [
        ("bit=0, basis='+'", "|0⟩",  "No gate — stays |0⟩",  C_BLUE),
        ("bit=1, basis='+'", "|1⟩",  "Pauli-X flip",          C_RED),
        ("bit=0, basis='x'", "|+⟩",  "Hadamard gate",         C_GREEN),
        ("bit=1, basis='x'", "|−⟩",  "Pauli-X → Hadamard",   C_ORANGE),
    ]

    for idx, (label, state, gate_desc, colour) in enumerate(circuits_info):
        y = 0.85 - idx * 0.22
        ax1.add_patch(plt.Rectangle((0.02, y - 0.08), 0.96, 0.18,
                                    transform=ax1.transAxes,
                                    facecolor=colour, alpha=0.12,
                                    edgecolor=colour, linewidth=1.5))
        ax1.text(0.06, y + 0.03, label,   transform=ax1.transAxes,
                 fontsize=9,  color='#2c3e50')
        ax1.text(0.06, y - 0.04, gate_desc, transform=ax1.transAxes,
                 fontsize=8,  color='#7f8c8d', style='italic')
        ax1.text(0.88, y - 0.01, state,   transform=ax1.transAxes,
                 fontsize=13, fontweight='bold', color=colour,
                 ha='center')

    # ═════════════════════════════════════════════════════════════════════════
    #  PANEL 2 – Basis comparison table (first 10 bits, no Eve run)
    # ═════════════════════════════════════════════════════════════════════════
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.axis('off')
    ax2.set_title('Basis Comparison — First 10 Bits (No Eve)',
                  fontweight='bold', fontsize=11, pad=8)

    n_show = min(10, res_no_eve['n_bits'])
    col_labels = ['Bit #', 'Alice\nBit', 'Alice\nBasis', 'Bob\nBasis', 'Bob\nResult', 'Match?']
    table_data = []
    for i in range(n_show):
        match = '✓' if res_no_eve['alice_bases'][i] == res_no_eve['bob_bases'][i] else '✗'
        table_data.append([
            str(i + 1),
            str(res_no_eve['alice_bits'][i]),
            res_no_eve['alice_bases'][i],
            res_no_eve['bob_bases'][i],
            str(res_no_eve['bob_results'][i]),
            match,
        ])

    tbl = ax2.table(
        cellText=table_data,
        colLabels=col_labels,
        cellLoc='center',
        loc='center',
        bbox=[0, 0, 1, 1],
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.5)

    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor('#bdc3c7')
        if r == 0:
            cell.set_facecolor('#2980b9')
            cell.set_text_props(color='white', fontweight='bold')
        else:
            match_col = table_data[r - 1][5]
            if match_col == '✓':
                cell.set_facecolor('#d5f5e3' if c < 5 else '#27ae60')
                if c == 5:
                    cell.set_text_props(color='white', fontweight='bold')
            else:
                cell.set_facecolor('#fdecea' if c < 5 else '#e74c3c')
                if c == 5:
                    cell.set_text_props(color='white', fontweight='bold')

    # ═════════════════════════════════════════════════════════════════════════
    #  PANEL 3 – Protocol statistics summary
    # ═════════════════════════════════════════════════════════════════════════
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.axis('off')
    ax3.set_title('Protocol Statistics Summary', fontweight='bold',
                  fontsize=11, pad=8)

    stats = [
        ('Total qubits sent',          res_no_eve['n_bits'],
         res_with_eve['n_bits'],        C_BLUE),
        ('Basis matches (sifted)',      len(res_no_eve['sifted_alice']),
         len(res_with_eve['sifted_alice']), C_GREEN),
        ('Sifting efficiency (%)',
         f"{len(res_no_eve['sifted_alice'])/res_no_eve['n_bits']*100:.1f}",
         f"{len(res_with_eve['sifted_alice'])/res_with_eve['n_bits']*100:.1f}",
         C_ORANGE),
        ('Bit errors detected',        res_no_eve['errors'],
         res_with_eve['errors'],        C_RED),
        ('QBER (%)',
         f"{res_no_eve['error_rate']:.1f}",
         f"{res_with_eve['error_rate']:.1f}",
         C_PURPLE),
    ]

    headers = ['Metric', 'No Eve', 'With Eve']
    stat_data = [[s[0], str(s[1]), str(s[2])] for s in stats]
    stat_colors = [[C_LIGHT, '#d5f5e3', '#fdecea']] * len(stats)

    tbl2 = ax3.table(
        cellText=stat_data,
        colLabels=headers,
        cellLoc='center',
        loc='center',
        bbox=[0, 0, 1, 1],
    )
    tbl2.auto_set_font_size(False)
    tbl2.set_fontsize(9)

    for (r, c), cell in tbl2.get_celld().items():
        cell.set_edgecolor('#bdc3c7')
        if r == 0:
            cell.set_facecolor('#34495e')
            cell.set_text_props(color='white', fontweight='bold')
        elif c == 0:
            cell.set_facecolor('#ecf0f1')
            cell.set_text_props(fontweight='bold')
        elif c == 1:
            cell.set_facecolor('#eafaf1')
        else:
            cell.set_facecolor('#fdedec')

    # ═════════════════════════════════════════════════════════════════════════
    #  PANEL 4 – Sifted key (No Eve)
    # ═════════════════════════════════════════════════════════════════════════
    ax4 = fig.add_subplot(gs[1, :2])
    ax4.set_title(f"Sifted Key Bits — No Eavesdropper "
                  f"(Key length = {len(res_no_eve['sifted_alice'])} bits)",
                  fontweight='bold', fontsize=11)

    n_key = len(res_no_eve['sifted_alice'])
    indices = list(range(n_key))
    alice_k = res_no_eve['sifted_alice']
    bob_k   = res_no_eve['sifted_bob']

    ax4.step(indices, alice_k, where='mid', color=C_BLUE,   lw=2.5,
             label="Alice's key", alpha=0.9)
    ax4.step(indices, [b + 0.05 for b in bob_k], where='mid',
             color=C_GREEN,  lw=1.5, linestyle='--',
             label="Bob's key  (offset +0.05 for visibility)", alpha=0.9)

    for i, (a, b) in enumerate(zip(alice_k, bob_k)):
        if a != b:
            ax4.axvspan(i - 0.5, i + 0.5, color=C_RED, alpha=0.3)

    ax4.set_yticks([0, 1])
    ax4.set_yticklabels(['0', '1'])
    ax4.set_xlabel('Key bit index', fontsize=9)
    ax4.set_ylabel('Bit value', fontsize=9)
    ax4.legend(fontsize=8, loc='upper right')
    ax4.set_xlim(-0.5, max(n_key - 0.5, 0.5))
    ax4.grid(axis='x', alpha=0.3, linestyle=':')
    ax4.set_facecolor('#f9f9f9')

    # ── binary key display ────────────────────────────────────────────────
    key_str = ''.join(map(str, alice_k))
    ax4.text(0.01, -0.18, f"Shared key: {key_str}",
             transform=ax4.transAxes, fontsize=8,
             fontfamily='monospace', color=C_BLUE,
             bbox=dict(boxstyle='round,pad=0.3', facecolor='#d6eaf8',
                       edgecolor=C_BLUE, linewidth=1.2))

    # ═════════════════════════════════════════════════════════════════════════
    #  PANEL 5 – Sifted key (With Eve) — errors highlighted
    # ═════════════════════════════════════════════════════════════════════════
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.set_title(f"Sifted Key — With Eve\n"
                  f"(QBER = {res_with_eve['error_rate']:.1f}%)",
                  fontweight='bold', fontsize=11)

    n_key_e = len(res_with_eve['sifted_alice'])
    alice_ke = res_with_eve['sifted_alice']
    bob_ke   = res_with_eve['sifted_bob']
    error_flags = [int(a != b) for a, b in zip(alice_ke, bob_ke)]

    ax5.bar(range(n_key_e), alice_ke,
            color=[C_RED if e else C_BLUE for e in error_flags],
            alpha=0.8, edgecolor='white', linewidth=0.5, label='Alice bit')
    ax5.bar(range(n_key_e), [-0.05] * n_key_e,
            bottom=bob_ke,
            color=[C_ORANGE if e else C_GREEN for e in error_flags],
            alpha=0.5, linewidth=0, label='Bob bit')

    ax5.set_yticks([0, 1])
    ax5.set_yticklabels(['0', '1'])
    ax5.set_xlabel('Key bit index', fontsize=9)
    ax5.set_ylabel('Bit value', fontsize=9)
    ax5.set_facecolor('#fff8f8')
    ax5.grid(axis='y', alpha=0.3, linestyle=':')

    error_count = sum(error_flags)
    ax5.text(0.5, 0.92,
             f"{error_count} error(s) detected → Eavesdropper present!",
             transform=ax5.transAxes, fontsize=8, ha='center',
             color=C_RED, fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='#fdecea',
                       edgecolor=C_RED, linewidth=1.2))

    # ═════════════════════════════════════════════════════════════════════════
    #  PANEL 6 – QBER bar chart
    # ═════════════════════════════════════════════════════════════════════════
    ax6 = fig.add_subplot(gs[2, 0])
    ax6.set_title('QBER: No Eve vs. With Eve', fontweight='bold', fontsize=11)

    scenarios = ['No Eve', 'With Eve']
    qbers     = [res_no_eve['error_rate'], res_with_eve['error_rate']]
    colours   = [C_GREEN, C_RED]
    bars      = ax6.bar(scenarios, qbers, color=colours, alpha=0.85,
                        edgecolor='black', linewidth=1.5, width=0.5)

    for bar, val in zip(bars, qbers):
        ax6.text(bar.get_x() + bar.get_width() / 2,
                 val + 0.5, f'{val:.1f}%',
                 ha='center', va='bottom', fontweight='bold', fontsize=11)

    ax6.axhline(y=11, color=C_ORANGE, linestyle='--', lw=2,
                label='~11% threshold\n(theoretical)')
    ax6.set_ylabel('Quantum Bit Error Rate (%)', fontsize=9)
    ax6.set_ylim(0, max(max(qbers) + 10, 30))
    ax6.legend(fontsize=8)
    ax6.set_facecolor('#f9f9f9')
    ax6.grid(axis='y', alpha=0.3)

    # ═════════════════════════════════════════════════════════════════════════
    #  PANEL 7 – Key length retention
    # ═════════════════════════════════════════════════════════════════════════
    ax7 = fig.add_subplot(gs[2, 1])
    ax7.set_title('Key Length Retention', fontweight='bold', fontsize=11)

    labels = ['Qubits\nSent', 'Sifted Key\n(No Eve)', 'Sifted Key\n(With Eve)']
    values = [
        res_no_eve['n_bits'],
        len(res_no_eve['sifted_alice']),
        len(res_with_eve['sifted_alice']),
    ]
    bar_colors = [C_BLUE, C_GREEN, C_RED]
    bars2 = ax7.bar(labels, values, color=bar_colors, alpha=0.85,
                    edgecolor='black', linewidth=1.2, width=0.55)

    for bar, val in zip(bars2, values):
        ax7.text(bar.get_x() + bar.get_width() / 2,
                 val + 0.3, str(val),
                 ha='center', va='bottom', fontweight='bold', fontsize=11)

    ax7.set_ylabel('Number of bits', fontsize=9)
    ax7.set_facecolor('#f9f9f9')
    ax7.grid(axis='y', alpha=0.3)

    # ═════════════════════════════════════════════════════════════════════════
    #  PANEL 8 – Protocol flow diagram
    # ═════════════════════════════════════════════════════════════════════════
    ax8 = fig.add_subplot(gs[2, 2])
    ax8.axis('off')
    ax8.set_title('BB84 Protocol Flow', fontweight='bold', fontsize=11)
    ax8.set_xlim(0, 10)
    ax8.set_ylim(0, 10)

    steps = [
        (5, 9.0, 'ALICE', C_BLUE,   True),
        (5, 7.5, '1. Generate random bits + bases', '#2c3e50', False),
        (5, 6.5, '2. Encode qubits (H / X gates)', '#2c3e50', False),
        (5, 5.5, '⟶  Quantum channel  ⟶', C_PURPLE, False),
        (5, 4.5, '3. Bob measures in random bases', '#2c3e50', False),
        (5, 3.5, '⟵  Classical channel ⟵', C_GREY, False),
        (5, 2.5, '4. Sifting: compare bases publicly', '#2c3e50', False),
        (5, 1.5, '5. Estimate QBER → detect Eve', '#2c3e50', False),
        (5, 0.5, 'BOB', C_GREEN,  True),
    ]

    for x, y, text, colour, bold in steps:
        ax8.text(x, y, text, ha='center', va='center',
                 fontsize=9, color=colour,
                 fontweight='bold' if bold else 'normal',
                 bbox=dict(boxstyle='round,pad=0.3',
                           facecolor=colour if bold else '#ecf0f1',
                           alpha=0.15 if not bold else 0.25,
                           edgecolor=colour, linewidth=1))

    # ── Bottom caption ────────────────────────────────────────────────────────
    fig.text(
        0.5, 0.01,
        'AIM3231 Emerging Tools & Technologies Lab  |  '
        'Qiskit Aer Simulator  |  BB84 QKD Protocol',
        ha='center', fontsize=9, color='#7f8c8d',
        style='italic'
    )

    plt.savefig(save_path, dpi=200, bbox_inches='tight',
                facecolor='white', pad_inches=0.2)
    print(f"\n✓ Visualization saved → {save_path}")
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 3: MAIN DRIVER
# ─────────────────────────────────────────────────────────────────────────────

def print_section(title: str):
    width = 70
    print()
    print('═' * width)
    print(f"  {title}")
    print('═' * width)


def print_protocol_table(res: dict, label: str):
    """Pretty-print the first 15 rows of the protocol table."""
    n = min(15, res['n_bits'])
    print(f"\n{'Bit':>4} | {'Alice':>5} | {'Alice':>6} | {'Bob':>6} | {'Bob':>6} | {'Match':>5} | {'Note'}")
    print(f"{'#':>4} | {'Bit':>5} | {'Basis':>6} | {'Basis':>6} | {'Result':>6} | {'':>5} |")
    print('-' * 60)
    for i in range(n):
        match    = res['alice_bases'][i] == res['bob_bases'][i]
        match_ch = '✓ KEEP' if match else '✗ DROP'
        error    = match and (res['alice_bits'][i] != res['bob_results'][i])
        note     = ' ← ERROR' if error else ''
        print(f"  {i+1:>2} | {res['alice_bits'][i]:>5} | {res['alice_bases'][i]:>6} | "
              f"{res['bob_bases'][i]:>6} | {res['bob_results'][i]:>6} | {match_ch} |{note}")


def main():
    random.seed(42)    # For reproducibility in demonstration

    N_BITS = 30        # Number of qubits to simulate

    print_section("AIM3231 MINI PROJECT — BB84 QUANTUM KEY DISTRIBUTION")
    print(f"  Manipal University Jaipur | Department of AI & ML")
    print(f"  Simulating {N_BITS} qubits using Qiskit Aer Simulator")

    # ── Run 1: No Eve ─────────────────────────────────────────────────────────
    print_section("RUN 1 — NO EAVESDROPPER (Ideal Secure Channel)")
    res_no_eve = run_bb84_protocol(n_bits=N_BITS, eavesdrop=False)
    print_protocol_table(res_no_eve, "No Eve")

    print(f"\n  Total qubits transmitted   : {res_no_eve['n_bits']}")
    print(f"  Basis matches (sifted key) : {len(res_no_eve['sifted_alice'])}")
    print(f"  Sifting efficiency         : "
          f"{len(res_no_eve['sifted_alice'])/N_BITS*100:.1f}%  (expected ≈ 50%)")
    print(f"  Bit errors in sifted key   : {res_no_eve['errors']}")
    print(f"  Quantum Bit Error Rate     : {res_no_eve['error_rate']:.2f}%  (expected ≈ 0%)")
    print(f"\n  Alice's sifted key  : {''.join(map(str, res_no_eve['sifted_alice']))}")
    print(f"  Bob's   sifted key  : {''.join(map(str, res_no_eve['sifted_bob']))}")
    keys_match = res_no_eve['sifted_alice'] == res_no_eve['sifted_bob']
    print(f"\n  ✓ Keys identical    : {keys_match}")
    if keys_match:
        print("  ✓ SECURE KEY ESTABLISHED — No eavesdropping detected.")
    else:
        print("  ✗ Key mismatch detected.")

    # ── Run 2: Eve present ────────────────────────────────────────────────────
    print_section("RUN 2 — WITH EAVESDROPPER (Eve intercepts every qubit)")
    res_with_eve = run_bb84_protocol(n_bits=N_BITS, eavesdrop=True)
    print_protocol_table(res_with_eve, "With Eve")

    print(f"\n  Total qubits transmitted   : {res_with_eve['n_bits']}")
    print(f"  Basis matches (sifted key) : {len(res_with_eve['sifted_alice'])}")
    print(f"  Bit errors in sifted key   : {res_with_eve['errors']}")
    print(f"  Quantum Bit Error Rate     : {res_with_eve['error_rate']:.2f}%  (expected ≈ 25%)")
    print(f"\n  Alice's sifted key  : {''.join(map(str, res_with_eve['sifted_alice']))}")
    print(f"  Bob's   sifted key  : {''.join(map(str, res_with_eve['sifted_bob']))}")

    if res_with_eve['error_rate'] > 10:
        print(f"\n  ✗ EAVESDROPPING DETECTED — QBER = {res_with_eve['error_rate']:.1f}% exceeds 11% threshold.")
        print("    Key is DISCARDED. Protocol restarts on a new channel.")
    else:
        print(f"\n  QBER = {res_with_eve['error_rate']:.1f}% — Below threshold (may be statistical fluke).")

    # ── Classical vs Quantum security comparison ──────────────────────────────
    print_section("QUANTUM ADVANTAGE — SECURITY ANALYSIS")
    print("""
  Classical Cryptography:
    • RSA-2048 relies on computational hardness (integer factorisation).
    • A sufficiently powerful quantum computer (Shor's algorithm) can
      break RSA in polynomial time.
    • Security is computational, NOT information-theoretic.

  BB84 Quantum Key Distribution:
    • Security is guaranteed by the laws of quantum mechanics.
    • No-Cloning Theorem: Eve cannot copy an unknown qubit.
    • Heisenberg Uncertainty: Measuring a qubit in the wrong basis
      disturbs it irreversibly → detectable as QBER ≈ 25%.
    • Even with unlimited computing power, Eve cannot intercept
      without leaving a detectable trace.
    • Information-theoretic security (unconditional security).

  Detection capability:
    • If Eve intercepts ALL qubits:  QBER ≈ 25%  → always detected.
    • If Eve intercepts 50% of qubits: QBER ≈ 12.5% → usually detected.
    • Standard abort threshold: QBER > 11%.
    """)

    # ── Visualization ─────────────────────────────────────────────────────────
    print_section("GENERATING VISUALIZATION")
    visualize_results(res_no_eve, res_with_eve, save_path='bb84_results.png')

    print_section("SIMULATION COMPLETE")
    print("  Output file : bb84_results.png")
    print("  To run      : python bb84_qkd_project.py")
    print("  Requirements: qiskit, qiskit-aer, numpy, matplotlib")
    print()


if __name__ == "__main__":
    main()
