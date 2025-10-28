"""Module for control system analysis and regulation."""
from typing import Union, List, Tuple, Optional
from functools import lru_cache
import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
from scipy import signal


class ControlSystem:
    """
    Class for control system analysis in automatic control.
    
    This class encapsulates a regulation system with its process and controller.
    
    Attributes:
        num: Numerator coefficients of Ga(s)
        den: Denominator coefficients of Ga(s)
        controller_type: Controller type ('P', 'PI', 'PD', 'PID')
        Kp: Proportional gain
        Ti: Integral time (for PI and PID)
        Td: Derivative time (for PD and PID)
    """
    
    # Constant for s->0 approximation in static error calculations
    _S_SMALL = 1e-10
    
    def __init__(
        self,
        num: Union[List[float], NDArray],
        den: Union[List[float], NDArray],
        controller_type: str,
        Kp: float,
        Ti: Optional[float] = None,
        Td: Optional[float] = None
    ) -> None:
        """
        Initializes a control system with its process and controller.
        
        Args:
            num: Numerator polynomial coefficients of Ga(s) (descending order)
            den: Denominator polynomial coefficients of Ga(s) (descending order)
            controller_type: Controller type ('P', 'PI', 'PD', 'PID')
            Kp: Proportional gain
            Ti: Integral time (required for PI and PID)
            Td: Derivative time (required for PD and PID)
            
        Raises:
            ValueError: If parameters are invalid for the controller type
            
        Example:
            >>> # PID controller
            >>> sys = ControlSystem([1], [1, 2, 1], 'PID', Kp=2.0, Ti=1.5, Td=0.5)
            >>> # PI controller
            >>> sys = ControlSystem([1], [1, 1], 'PI', Kp=1.0, Ti=2.0)
        """
        # Store system coefficients
        self.num = np.asarray(num)
        self.den = np.asarray(den)
        
        # Validate denominator
        if len(self.den) == 0 or np.allclose(self.den, 0):
            raise ValueError("Denominator cannot be empty or zero")
        
        # Store controller type
        self.controller_type = controller_type.upper()
        
        # Validate and store controller parameters
        self.Kp = Kp
        self.Ti = Ti
        self.Td = Td
        
        # Cache for controller coefficients
        self._gc_coeffs_cache: Optional[Tuple[NDArray, NDArray]] = None
        
        # Validate according to controller type
        self._validate_controller_params()
    
    def _validate_controller_params(self) -> None:
        """Validates controller parameters according to its type."""
        if self.controller_type == 'P':
            pass  # Only Kp is required
            
        elif self.controller_type == 'PI':
            if self.Ti is None:
                raise ValueError("PI controller requires Ti parameter")
            if self.Ti == 0:
                raise ValueError("Ti cannot be zero")
                
        elif self.controller_type == 'PD':
            if self.Td is None:
                raise ValueError("PD controller requires Td parameter")
                
        elif self.controller_type == 'PID':
            if self.Ti is None or self.Td is None:
                raise ValueError("PID controller requires Ti and Td parameters")
            if self.Ti == 0:
                raise ValueError("Ti cannot be zero")
                
        else:
            raise ValueError(
                f"Controller type '{self.controller_type}' not supported. "
                "Valid types: 'P', 'PI', 'PD', 'PID'"
            )
    
    def _get_gc_coeffs(self) -> Tuple[NDArray, NDArray]:
        """
        Gets the polynomial coefficients of the controller Gc(s).
        
        Returns:
            Tuple (num, den): Numerator and denominator of Gc(s)
        """
        # Use cache if available
        if self._gc_coeffs_cache is not None:
            return self._gc_coeffs_cache
        
        if self.controller_type == 'P':
            result = np.array([self.Kp]), np.array([1.0])
            
        elif self.controller_type == 'PI':
            assert self.Ti is not None
            # Gc(s) = Kp * (Ti*s + 1) / (Ti*s)
            num = np.array([self.Kp * self.Ti, self.Kp])
            den = np.array([self.Ti, 0.0])
            result = num, den
            
        elif self.controller_type == 'PD':
            assert self.Td is not None
            # Gc(s) = Kp * (Td*s + 1)
            num = np.array([self.Kp * self.Td, self.Kp])
            den = np.array([1.0])
            result = num, den
            
        else:  # PID
            assert self.Ti is not None and self.Td is not None
            # Gc(s) = Kp * (Ti*Td*s^2 + Ti*s + 1) / (Ti*s)
            num = np.array([self.Kp * self.Ti * self.Td, self.Kp * self.Ti, self.Kp])
            den = np.array([self.Ti, 0.0])
            result = num, den
        
        # Cache the result
        self._gc_coeffs_cache = result
        return result
    
    @staticmethod
    def _maximize_plot_window() -> None:
        """Maximizes the matplotlib window if possible."""
        manager = plt.get_current_fig_manager()
        try:
            window = getattr(manager, 'window', None)
            if window is not None:
                if hasattr(window, 'state'):
                    window.state('zoomed')  # type: ignore
                elif hasattr(window, 'showMaximized'):
                    window.showMaximized()  # type: ignore
        except Exception:
            pass  # If it doesn't work, ignore
    
    @staticmethod
    def _interpolate_crossing(x_data: NDArray, y_data: NDArray, y_target: float) -> Optional[float]:
        """
        Interpolates to find where y_data crosses y_target.
        
        Args:
            x_data: X data
            y_data: Y data
            y_target: Target value to cross
            
        Returns:
            x value at crossing, or None if no crossing
        """
        # Check if target is in range
        if (np.min(y_data) <= y_target <= np.max(y_data)) or (np.max(y_data) <= y_target <= np.min(y_data)):
            return float(np.interp(y_target, y_data[::-1], x_data[::-1]))
        return None
    
    def _compute_step_characteristics(self, t: NDArray, y: NDArray) -> dict:
        """
        Computes step response characteristics.
        
        Args:
            t: Time
            y: Response
            
        Returns:
            Dictionary with computed characteristics
        """
        y_final = y[-1]
        characteristics = {
            'y_final': y_final,
            't_rise': None,
            't_settle': None,
            'overshoot': 0.0
        }
        
        # Rise time (10% to 90%)
        idx_10 = np.argmax(y >= 0.1 * y_final)
        idx_90 = np.argmax(y >= 0.9 * y_final)
        if idx_90 > idx_10:
            characteristics['t_rise'] = t[idx_90] - t[idx_10]
        
        # Overshoot
        y_max = np.max(y)
        if y_max > y_final and y_final > 0:
            characteristics['overshoot'] = ((y_max - y_final) / y_final) * 100
        
        # Settling time (5%)
        tolerance = 0.05 * abs(y_final)
        settled = np.abs(y - y_final) <= tolerance
        if np.any(settled):
            characteristics['t_settle'] = t[np.where(settled)[0][0]]
        
        return characteristics
    
    @property
    def poles_open_loop(self) -> NDArray:
        """
        Calculates the open-loop system Go(s) poles.
        
        Returns:
            Array of poles
        """
        num_gc, den_gc = self._get_gc_coeffs()
        den_go = np.polymul(self.den, den_gc)
        return np.roots(den_go)
    
    @property
    def zeros_open_loop(self) -> NDArray:
        """
        Calculates the open-loop system Go(s) zeros.
        
        Returns:
            Array of zeros
        """
        num_gc, den_gc = self._get_gc_coeffs()
        num_go = np.polymul(self.num, num_gc)
        return np.roots(num_go)
    
    @property
    def poles_closed_loop(self) -> NDArray:
        """
        Calculates the closed-loop system Gf(s) poles.
        
        Returns:
            Array of poles
        """
        num_gc, den_gc = self._get_gc_coeffs()
        num_go = np.polymul(self.num, num_gc)
        den_go = np.polymul(self.den, den_gc)
        den_gf = np.polyadd(den_go, num_go)
        return np.roots(den_gf)
    
    @property
    def zeros_closed_loop(self) -> NDArray:
        """
        Calculates the closed-loop system Gf(s) zeros.
        
        Returns:
            Array of zeros
        """
        num_gc, den_gc = self._get_gc_coeffs()
        num_gf = np.polymul(self.num, num_gc)
        return np.roots(num_gf)
    
    def Ga(self, s: Union[complex, NDArray, None] = None) -> Union[complex, NDArray]:
        """
        Calculates the process transfer function Ga(s) = num(s) / den(s).
        
        Args:
            s: Laplace variable (default: 1j for frequency response)
            
        Returns:
            Process transfer function value(s)
        """
        if s is None:
            s = 1j
        
        return np.polyval(self.num, s) / np.polyval(self.den, s)
    
    def Gc(self, s: Union[complex, NDArray, None] = None) -> Union[float, complex, NDArray]:
        """
        Calculates the controller transfer function Gc(s).
        
        Args:
            s: Laplace variable (default: 1j for frequency response)
            
        Returns:
            Controller transfer function value(s)
        """
        if s is None:
            s = 1j
        
        if self.controller_type == 'P':
            return self.Kp
            
        elif self.controller_type == 'PI':
            assert self.Ti is not None, "Ti should not be None for PI controller"
            return self.Kp * (1 + 1 / (self.Ti * s))
            
        elif self.controller_type == 'PD':
            assert self.Td is not None, "Td should not be None for PD controller"
            return self.Kp * (1 + self.Td * s)
            
        else:  # PID
            assert self.Ti is not None and self.Td is not None, "Ti and Td should not be None for PID controller"
            return self.Kp * (1 + 1 / (self.Ti * s) + self.Td * s)
    
    def Go(self, s: Union[complex, NDArray, None] = None) -> Union[complex, NDArray]:
        """
        Calculates the open-loop transfer function Go(s) = Ga(s) * Gc(s).
        
        Args:
            s: Laplace variable (default: 1j for frequency response)
            
        Returns:
            Open-loop transfer function
        """
        return self.Ga(s) * self.Gc(s)
    
    def Gf(self, s: Union[complex, NDArray, None] = None) -> Union[complex, NDArray]:
        """
        Calculates the closed-loop transfer function Gf(s) = Go(s) / (1 + Go(s)).
        
        Args:
            s: Laplace variable (default: 1j for frequency response)
            
        Returns:
            Closed-loop transfer function
        """
        Go_val = self.Go(s)
        return Go_val / (1 + Go_val)
    
    def Ge(self, s: Union[complex, NDArray, None] = None) -> Union[complex, NDArray]:
        """
        Calculates the error transfer function Ge(s) = 1 / (1 + Go(s)).
        
        Args:
            s: Laplace variable (default: 1j for frequency response)
            
        Returns:
            Error transfer function
        """
        Go_val = self.Go(s)
        return 1 / (1 + Go_val)
    
    def bode(self, omega: NDArray, function: str = 'Go') -> Tuple[NDArray, NDArray]:
        """
        Calculates Bode diagram data.
        
        Args:
            omega: Frequency array (rad/s)
            function: Function to analyze ('Go', 'Ga', 'Gc', 'Gf', 'Ge') [default: 'Go']
            
        Returns:
            Tuple (magnitude_db, phase_deg):
                - magnitude_db: Magnitude in decibels
                - phase_deg: Phase in degrees
        """
        s_values = 1j * omega
        
        # Select transfer function
        func_upper = function.upper()
        if func_upper == 'GO':
            H = self.Go(s_values)
        elif func_upper == 'GA':
            H = self.Ga(s_values)
        elif func_upper == 'GC':
            H = self.Gc(s_values)
        elif func_upper == 'GF':
            H = self.Gf(s_values)
        elif func_upper == 'GE':
            H = self.Ge(s_values)
        else:
            raise ValueError(f"Function '{function}' not supported. Use 'Go', 'Ga', 'Gc', 'Gf', or 'Ge'")
        
        magnitude_db = 20 * np.log10(np.abs(H))
        phase_deg = np.angle(H, deg=True)
        
        return magnitude_db, phase_deg
    
    def margin(self, omega: NDArray) -> Tuple[float, float, float, float]:
        """
        Calculates stability margins (gain and phase) for the open-loop system.
        
        Args:
            omega: Frequency array (rad/s)
            
        Returns:
            Tuple (gain_margin_db, phase_margin_deg, wgc, wpc):
                - gain_margin_db: Gain margin in dB
                - phase_margin_deg: Phase margin in degrees
                - wgc: Gain crossover frequency (rad/s)
                - wpc: Phase crossover frequency (rad/s)
        """
        s_values = 1j * omega
        H = self.Go(s_values)
        
        mag = np.abs(H)
        phase = np.angle(H, deg=True)
        mag_db = 20 * np.log10(mag)
        
        # Phase margin: phase at |H| = 1 (0 dB)
        wgc_interp = self._interpolate_crossing(omega, mag_db, 0.0)
        if wgc_interp is not None:
            wgc = wgc_interp
            phase_at_wgc = np.interp(wgc, omega, phase)
            phase_margin = 180 + phase_at_wgc
        else:
            idx_gain = np.argmin(np.abs(mag - 1))
            phase_margin = 180 + phase[idx_gain]
            wgc = omega[idx_gain]
        
        # Gain margin: 1/|H| at phase = -180°
        wpc_interp = self._interpolate_crossing(omega, phase, -180.0)
        if wpc_interp is not None:
            wpc = wpc_interp
            mag_at_wpc = np.interp(wpc, omega, mag)
            gain_margin_db = -20 * np.log10(mag_at_wpc)
        else:
            idx_phase = np.argmin(np.abs(phase + 180))
            gain_margin_db = -20 * np.log10(mag[idx_phase])
            wpc = omega[idx_phase]
        
        return gain_margin_db, phase_margin, wgc, wpc
    
    def step_response(self, t: Optional[NDArray] = None, function: str = 'Gf') -> Tuple[NDArray, NDArray]:
        """
        Calculates the system step response.
        
        Args:
            t: Time array (s). If None, generates automatically
            function: Function to analyze ('Gf' or 'Ge') [default: 'Gf']
            
        Returns:
            Tuple (t, y):
                - t: Time (s)
                - y: System time response
        """
        func_upper = function.upper()
        
        # Get closed-loop transfer function coefficients
        if func_upper == 'GF':
            # Calculate Gf = Go / (1 + Go) = Ga*Gc / (1 + Ga*Gc)
            num_gc, den_gc = self._get_gc_coeffs()
            
            # Go = Ga * Gc
            num_go = np.polymul(self.num, num_gc)
            den_go = np.polymul(self.den, den_gc)
            
            # Gf = Go / (1 + Go) = num_go / (den_go + num_go)
            num_gf = num_go
            den_gf = np.polyadd(den_go, num_go)
            
        elif func_upper == 'GE':
            # Calculate Ge = 1 / (1 + Go) = den_go / (den_go + num_go)
            num_gc, den_gc = self._get_gc_coeffs()
            
            num_go = np.polymul(self.num, num_gc)
            den_go = np.polymul(self.den, den_gc)
            
            num_gf = den_go
            den_gf = np.polyadd(den_go, num_go)
        else:
            raise ValueError(f"Function '{function}' not supported. Use 'Gf' or 'Ge'")
        
        # Create transfer system
        sys = signal.TransferFunction(num_gf, den_gf)
        
        # Calculate step response
        if t is None:
            t_out, y_out = signal.step(sys)
        else:
            t_out, y_out = signal.step(sys, T=t)
        
        return t_out, y_out
    
    def nyquist(self, omega: Optional[NDArray] = None) -> Tuple[NDArray, NDArray]:
        """
        Calculates Nyquist diagram data for Go(s).
        
        Args:
            omega: Frequency array (rad/s). If None, generates automatically
            
        Returns:
            Tuple (real, imag): Real and imaginary parts of Go(jω)
        """
        if omega is None:
            omega = np.logspace(-3, 3, 1000)
        
        s_values = 1j * omega
        H = self.Go(s_values)
        
        return np.real(H), np.imag(H)
    
    def plot_bode(self, omega: Optional[NDArray] = None, function: str = 'Go', 
                  show_margins: bool = True, figsize: Tuple[float, float] = (10, 8)) -> None:
        """
        Plots Bode diagram with optional margin annotations.
        
        Args:
            omega: Frequency array (rad/s). If None, generates automatically
            function: Function to plot ('Go', 'Ga', 'Gc', 'Gf', 'Ge')
            show_margins: If True, displays stability margins (only for 'Go')
            figsize: Figure size (width, height)
        """
        if omega is None:
            omega = np.logspace(-2, 3, 2000)
        
        mag, phase = self.bode(omega, function=function)
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)
        
        # Maximize window
        self._maximize_plot_window()
        
        # Magnitude diagram
        ax1.semilogx(omega, mag, 'b-', linewidth=1.5)
        ax1.grid(True, which='both', alpha=0.3)
        ax1.set_ylabel('Magnitude (dB)', fontsize=11)
        ax1.set_title(f'Bode Diagram of {function}(s)', fontsize=12, fontweight='bold')
        ax1.axhline(y=0, color='r', linestyle='--', alpha=0.5, linewidth=1)
        
        # Phase diagram
        ax2.semilogx(omega, phase, 'b-', linewidth=1.5)
        ax2.grid(True, which='both', alpha=0.3)
        ax2.set_xlabel('Frequency (rad/s)', fontsize=11)
        ax2.set_ylabel('Phase (degrees)', fontsize=11)
        ax2.axhline(y=-180, color='r', linestyle='--', alpha=0.5, linewidth=1)
        
        # Margin annotations (only for Go)
        func_upper = function.upper()
        if show_margins and func_upper == 'GO':
            gm, pm, wgc, wpc = self.margin(omega)
            
            # Mark ω_co (gain crossover frequency)
            ax1.axvline(x=wgc, color='g', linestyle=':', alpha=0.7, linewidth=2)
            ax1.plot(wgc, 0, 'go', markersize=8)
            ax1.text(wgc * 1.2, 5, f'ωco = {wgc:.2f} rad/s', fontsize=9, color='green')
            
            # Mark φ_m (phase margin)
            phase_at_wgc = np.interp(wgc, omega, phase)
            ax2.axvline(x=wgc, color='g', linestyle=':', alpha=0.7, linewidth=2)
            ax2.plot(wgc, phase_at_wgc, 'go', markersize=8)
            ax2.text(wgc * 1.2, phase_at_wgc + 10, f'φm = {pm:.2f}°', fontsize=9, color='green')
            ax2.plot([wgc, wgc], [-180, phase_at_wgc], 'g--', linewidth=1.5, alpha=0.7)
            
            # Mark ω_pc (phase crossover frequency)
            if wpc < omega[-1]:
                mag_at_wpc = np.interp(wpc, omega, mag)
                ax1.axvline(x=wpc, color='orange', linestyle=':', alpha=0.7, linewidth=2)
                ax1.plot(wpc, mag_at_wpc, 'o', color='orange', markersize=8)
                ax1.text(wpc * 1.2, mag_at_wpc - 10, f'Gm = {gm:.2f} dB', fontsize=9, color='orange')
                ax1.plot([wpc, wpc], [0, mag_at_wpc], 'orange', linestyle='--', linewidth=1.5, alpha=0.7)
                
                ax2.axvline(x=wpc, color='orange', linestyle=':', alpha=0.7, linewidth=2)
                ax2.plot(wpc, -180, 'o', color='orange', markersize=8)
        
        plt.tight_layout()
        plt.show()
    
    def plot_nyquist(self, omega: Optional[NDArray] = None, 
                     figsize: Tuple[float, float] = (8, 8)) -> None:
        """
        Plots Nyquist diagram of Go(s).
        
        Args:
            omega: Frequency array (rad/s). If None, generates automatically
            figsize: Figure size (width, height)
        """
        real, imag = self.nyquist(omega)
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Maximize window
        self._maximize_plot_window()
        
        # Plot Nyquist curve
        ax.plot(real, imag, 'b-', linewidth=1.5, label='ω > 0')
        ax.plot(real, -imag, 'b--', linewidth=1.5, alpha=0.5, label='ω < 0')
        
        # Critical point (-1, 0)
        ax.plot(-1, 0, 'rx', markersize=15, markeredgewidth=3, label='Critical point (-1, 0)')
        
        # Unit circle
        theta = np.linspace(0, 2*np.pi, 100)
        ax.plot(np.cos(theta), np.sin(theta), 'r--', alpha=0.3, linewidth=1)
        
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='k', linewidth=0.5)
        ax.axvline(x=0, color='k', linewidth=0.5)
        ax.set_xlabel('Real part', fontsize=11)
        ax.set_ylabel('Imaginary part', fontsize=11)
        ax.set_title('Nyquist Diagram of Go(s)', fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.axis('equal')
        
        plt.tight_layout()
        plt.show()
    
    def plot_step_response(self, t: Optional[NDArray] = None, function: str = 'Gf',
                          figsize: Tuple[float, float] = (10, 6)) -> None:
        """
        Plots system step response.
        
        Args:
            t: Time array (s). If None, generates automatically
            function: Function to analyze ('Gf' or 'Ge')
            figsize: Figure size (width, height)
        """
        t_out, y_out = self.step_response(t, function)
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Maximize window
        self._maximize_plot_window()
        
        ax.plot(t_out, y_out, 'b-', linewidth=2)
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('Time (s)', fontsize=11)
        ax.set_ylabel('Amplitude', fontsize=11)
        ax.set_title(f'Step Response of {function}(s)', fontsize=12, fontweight='bold')
        
        # Reference line (final value)
        func_upper = function.upper()
        if func_upper == 'GF':
            ax.axhline(y=1, color='r', linestyle='--', alpha=0.5, label='Setpoint')
        
        # Calculate and display characteristics
        if func_upper == 'GF':
            characteristics = self._compute_step_characteristics(t_out, y_out)
            
            # Display characteristics
            info_text = f'Final value: {characteristics["y_final"]:.3f}\n'
            if characteristics['t_rise'] is not None:
                info_text += f'Rise time: {characteristics["t_rise"]:.3f} s\n'
            if characteristics['overshoot'] > 0:
                info_text += f'Overshoot: {characteristics["overshoot"]:.1f}%\n'
            if characteristics['t_settle'] is not None:
                info_text += f'Settling time (5%): {characteristics["t_settle"]:.3f} s'
            
            ax.text(0.02, 0.98, info_text, transform=ax.transAxes, 
                    fontsize=9, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        ax.legend(fontsize=9)
        plt.tight_layout()
        plt.show()
    
    def static_error(self, input_type: str = 'step') -> float:
        """
        Calculates static error for different input types.
        
        Args:
            input_type: Input type ('step', 'ramp', 'parabola')
            
        Returns:
            Static error (can be inf if system doesn't follow)
        """
        input_lower = input_type.lower()
        
        # Calculate error constants
        if input_lower == 'step':
            # Kp = lim(s->0) Go(s)
            Kp = np.abs(self.Go(self._S_SMALL))
            return float(1 / (1 + Kp))
            
        elif input_lower == 'ramp':
            # Kv = lim(s->0) s*Go(s)
            Kv = np.abs(self._S_SMALL * self.Go(self._S_SMALL))
            if Kv == 0:
                return np.inf
            return float(1 / Kv)
            
        elif input_lower == 'parabola':
            # Ka = lim(s->0) s^2*Go(s)
            Ka = np.abs(self._S_SMALL**2 * self.Go(self._S_SMALL))
            if Ka == 0:
                return np.inf
            return float(1 / Ka)
            
        else:
            raise ValueError(f"Input type '{input_type}' not supported. Use 'step', 'ramp', or 'parabola'")
    
    def plot_all(self, omega: Optional[NDArray] = None, t: Optional[NDArray] = None,
                 figsize: Tuple[float, float] = (14, 10)) -> None:
        """
        Plots all diagrams in a single window (Bode, Nyquist, Step Response, Pole-Zero).
        
        Args:
            omega: Frequency array (rad/s). If None, generates automatically
            t: Time array (s). If None, generates automatically
            figsize: Figure size (width, height)
        """
        if omega is None:
            omega = np.logspace(-2, 3, 2000)
        
        fig = plt.figure(figsize=figsize)
        # Maximize window
        self._maximize_plot_window()
        
        gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
        
        # ========== Bode - Magnitude ==========
        ax1 = fig.add_subplot(gs[0, :])
        mag, phase = self.bode(omega, function='Go')
        ax1.semilogx(omega, mag, 'b-', linewidth=1.5)
        ax1.grid(True, which='both', alpha=0.3)
        ax1.set_ylabel('Magnitude (dB)', fontsize=10)
        ax1.set_title('Bode Diagram of Go(s)', fontsize=11, fontweight='bold')
        ax1.axhline(y=0, color='r', linestyle='--', alpha=0.5, linewidth=1)
        
        # Margin annotations
        gm, pm, wgc, wpc = self.margin(omega)
        ax1.axvline(x=wgc, color='g', linestyle=':', alpha=0.7, linewidth=2)
        ax1.plot(wgc, 0, 'go', markersize=7)
        ax1.text(wgc * 1.2, 5, f'ωco = {wgc:.2f} rad/s', fontsize=8, color='green')
        
        if wpc < omega[-1]:
            mag_at_wpc = np.interp(wpc, omega, mag)
            ax1.axvline(x=wpc, color='orange', linestyle=':', alpha=0.7, linewidth=2)
            ax1.plot(wpc, mag_at_wpc, 'o', color='orange', markersize=7)
            ax1.text(wpc * 1.2, mag_at_wpc - 10, f'Gm = {gm:.2f} dB', fontsize=8, color='orange')
        
        # ========== Bode - Phase ==========
        ax2 = fig.add_subplot(gs[1, :])
        ax2.semilogx(omega, phase, 'b-', linewidth=1.5)
        ax2.grid(True, which='both', alpha=0.3)
        ax2.set_xlabel('Frequency (rad/s)', fontsize=10)
        ax2.set_ylabel('Phase (degrees)', fontsize=10)
        ax2.axhline(y=-180, color='r', linestyle='--', alpha=0.5, linewidth=1)
        
        # Annotations
        phase_at_wgc = np.interp(wgc, omega, phase)
        ax2.axvline(x=wgc, color='g', linestyle=':', alpha=0.7, linewidth=2)
        ax2.plot(wgc, phase_at_wgc, 'go', markersize=7)
        ax2.text(wgc * 1.2, phase_at_wgc + 10, f'φm = {pm:.2f}°', fontsize=8, color='green')
        ax2.plot([wgc, wgc], [-180, phase_at_wgc], 'g--', linewidth=1.5, alpha=0.7)
        
        if wpc < omega[-1]:
            ax2.axvline(x=wpc, color='orange', linestyle=':', alpha=0.7, linewidth=2)
            ax2.plot(wpc, -180, 'o', color='orange', markersize=7)
        
        # ========== Nyquist ==========
        ax3 = fig.add_subplot(gs[2, 0])
        real, imag = self.nyquist(omega)
        ax3.plot(real, imag, 'b-', linewidth=1.5, label='ω > 0')
        ax3.plot(real, -imag, 'b--', linewidth=1.5, alpha=0.5, label='ω < 0')
        ax3.plot(-1, 0, 'rx', markersize=12, markeredgewidth=2.5, label='Critical point')
        
        # Unit circle
        theta = np.linspace(0, 2*np.pi, 100)
        ax3.plot(np.cos(theta), np.sin(theta), 'r--', alpha=0.3, linewidth=1)
        
        ax3.grid(True, alpha=0.3)
        ax3.axhline(y=0, color='k', linewidth=0.5)
        ax3.axvline(x=0, color='k', linewidth=0.5)
        ax3.set_xlabel('Real part', fontsize=10)
        ax3.set_ylabel('Imaginary part', fontsize=10)
        ax3.set_title('Nyquist Diagram', fontsize=11, fontweight='bold')
        ax3.legend(fontsize=7, loc='best')
        ax3.axis('equal')
        
        # ========== Step Response ==========
        ax4 = fig.add_subplot(gs[2, 1])
        t_out, y_out = self.step_response(t, function='Gf')
        ax4.plot(t_out, y_out, 'b-', linewidth=2)
        ax4.grid(True, alpha=0.3)
        ax4.set_xlabel('Time (s)', fontsize=10)
        ax4.set_ylabel('Amplitude', fontsize=10)
        ax4.set_title('Step Response of Gf(s)', fontsize=11, fontweight='bold')
        ax4.axhline(y=1, color='r', linestyle='--', alpha=0.5, label='Setpoint')
        
        # Calculate and display characteristics
        characteristics = self._compute_step_characteristics(t_out, y_out)
        
        # Display characteristics
        info_text = f'Final value: {characteristics["y_final"]:.3f}\n'
        if characteristics['t_rise'] is not None:
            info_text += f'Rise time: {characteristics["t_rise"]:.3f} s\n'
        if characteristics['overshoot'] > 0:
            info_text += f'Overshoot: {characteristics["overshoot"]:.1f}%\n'
        if characteristics['t_settle'] is not None:
            info_text += f'Settling time (5%): {characteristics["t_settle"]:.3f} s'
        
        ax4.text(0.98, 0.02, info_text, transform=ax4.transAxes, 
                fontsize=8, verticalalignment='bottom', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        ax4.legend(fontsize=7, loc='upper left')
        
        plt.suptitle(f'Complete System Analysis - {self.controller_type} (Kp={self.Kp:.2f}' + 
                    (f', Ti={self.Ti}' if self.Ti is not None else '') + 
                    (f', Td={self.Td}' if self.Td is not None else '') + ')',
                    fontsize=13, fontweight='bold', y=0.995)
        
        plt.show()
    
    def __repr__(self) -> str:
        """String representation of the system."""
        return (
            f"ControlSystem(\n"
            f"  Ga(s) = {self.num.tolist()} / {self.den.tolist()}\n"
            f"  Controller: {self.controller_type}\n"
            f"  Kp={self.Kp}" +
            (f", Ti={self.Ti}" if self.Ti is not None else "") +
            (f", Td={self.Td}" if self.Td is not None else "") +
            "\n)"
        )