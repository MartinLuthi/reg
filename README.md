# Control System Analysis

A Python library for analyzing and visualizing control systems (automatic regulation). This library provides comprehensive tools for analyzing feedback control systems with various controller types (P, PI, PD, PID).

## Features

- **Multiple Controller Types**: Support for P, PI, PD, and PID controllers
- **Complete System Analysis**:
  - Bode diagrams (magnitude and phase)
  - Nyquist diagrams
  - Step response analysis
  - Pole-zero analysis
  - Stability margins (gain and phase margins)
  - Static error calculations
- **Visualization Tools**: Professional plotting functions with matplotlib
- **Transfer Functions**: Calculate open-loop (Go), closed-loop (Gf), and error (Ge) transfer functions

## Installation

### Requirements

- Python 3.8+
- NumPy
- SciPy
- Matplotlib

### Install Dependencies

```bash
pip install numpy scipy matplotlib
```

## Quick Start

```python
import numpy as np
from ControlSystem import ControlSystem

# Define process transfer function Ga(s) = 100/((s+10)(s+20))
# Which equals: 100/(s^2 + 30s + 200)
Ga_num = [100.0]
Ga_den = [1.0, 30.0, 200.0]

# Create a PI controller system
Kp = 4 * np.sqrt(2)
Ti = 0.1

system = ControlSystem(
    num=Ga_num,
    den=Ga_den,
    controller_type='PI',
    Kp=Kp,
    Ti=Ti
)

# Analyze stability margins
omega = np.logspace(-1, 3, 2000)
gain_margin, phase_margin, wgc, wpc = system.margin(omega)

print(f"Gain Margin: {gain_margin:.2f} dB")
print(f"Phase Margin: {phase_margin:.2f}°")

# Plot all analysis diagrams
system.plot_all(omega)
```

## Usage Examples

### Creating Different Controller Types

#### Proportional (P) Controller
```python
system_p = ControlSystem([1], [1, 2, 1], 'P', Kp=2.0)
```

#### Proportional-Integral (PI) Controller
```python
system_pi = ControlSystem([1], [1, 1], 'PI', Kp=1.0, Ti=2.0)
```

#### Proportional-Derivative (PD) Controller
```python
system_pd = ControlSystem([1], [1, 1], 'PD', Kp=1.0, Td=0.5)
```

#### Proportional-Integral-Derivative (PID) Controller
```python
system_pid = ControlSystem([1], [1, 2, 1], 'PID', Kp=2.0, Ti=1.5, Td=0.5)
```

### System Analysis Methods

#### Transfer Functions
```python
# Open-loop transfer function Go(s) = Ga(s) * Gc(s)
Go = system.Go(1j * omega)

# Closed-loop transfer function Gf(s) = Go(s) / (1 + Go(s))
Gf = system.Gf(1j * omega)

# Error transfer function Ge(s) = 1 / (1 + Go(s))
Ge = system.Ge(1j * omega)
```

#### Poles and Zeros
```python
print("Open-loop poles:", system.poles_open_loop)
print("Open-loop zeros:", system.zeros_open_loop)
print("Closed-loop poles:", system.poles_closed_loop)
print("Closed-loop zeros:", system.zeros_closed_loop)
```

#### Static Error Analysis
```python
# Static error for step input
error_step = system.static_error('step')

# Static error for ramp input
error_ramp = system.static_error('ramp')

# Static error for parabolic input
error_parabola = system.static_error('parabola')
```

#### Individual Plot Functions
```python
# Bode diagram
system.plot_bode(omega, function='Go', show_margins=True)

# Nyquist diagram
system.plot_nyquist(omega)

# Step response
system.plot_step_response()

# All diagrams in one window
system.plot_all(omega)
```

## API Reference

### `ControlSystem` Class

#### Constructor
```python
ControlSystem(num, den, controller_type, Kp, Ti=None, Td=None)
```

**Parameters:**
- `num`: Numerator coefficients of Ga(s) (descending order)
- `den`: Denominator coefficients of Ga(s) (descending order)
- `controller_type`: Controller type ('P', 'PI', 'PD', 'PID')
- `Kp`: Proportional gain
- `Ti`: Integral time (required for PI and PID)
- `Td`: Derivative time (required for PD and PID)

#### Key Methods

| Method | Description | Returns |
|--------|-------------|---------|
| `Ga(s)` | Process transfer function | Complex/array |
| `Gc(s)` | Controller transfer function | Complex/array |
| `Go(s)` | Open-loop transfer function | Complex/array |
| `Gf(s)` | Closed-loop transfer function | Complex/array |
| `Ge(s)` | Error transfer function | Complex/array |
| `bode(omega, function)` | Bode diagram data | (magnitude_db, phase_deg) |
| `margin(omega)` | Stability margins | (gain_margin, phase_margin, wgc, wpc) |
| `step_response(t, function)` | Step response | (time, response) |
| `nyquist(omega)` | Nyquist data | (real, imag) |
| `static_error(input_type)` | Static error | float |
| `plot_bode()` | Plot Bode diagram | None |
| `plot_nyquist()` | Plot Nyquist diagram | None |
| `plot_step_response()` | Plot step response | None |
| `plot_all()` | Plot all diagrams | None |

#### Properties

- `poles_open_loop`: Open-loop poles
- `zeros_open_loop`: Open-loop zeros
- `poles_closed_loop`: Closed-loop poles
- `zeros_closed_loop`: Closed-loop zeros

## Example: Complete System Analysis

See `Exo_phi_m.py` for a complete example that demonstrates:
- System configuration with a PI controller
- Pole and zero analysis
- Static error calculations
- Stability margin computations
- Comprehensive visualization

Run the example:
```bash
python Exo_phi_m.py
```

## Project Structure

```
reg/
├── ControlSystem.py    # Main control system library
├── Exo_phi_m.py       # Example: PI controller analysis
├── README.md          # This file
├── LICENSE            # License information
└── .gitignore         # Git ignore rules
```

## License

See [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Author

MartinLuthi

## Acknowledgments

This library is designed for educational purposes and control systems engineering applications.
