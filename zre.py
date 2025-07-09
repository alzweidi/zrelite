#!/usr/bin/env python3
"""
ZAWAD RISK ENGINE v3.9 (ZRE)
============================
Multi-Level Take Profit Position Sizing System with Enhanced Kelly Criterion

This system implements a mathematically rigorous approach to position sizing using the Kelly
Criterion with weighted R-ratios for multi-level take profit strategies, designed for
institutional capital preservation and optimal risk management.

# MATHEMATICAL FOUNDATION:
# ========================
# Enhanced Kelly Criterion: k = (p × (Weighted_R + 1) - 1) / Weighted_R
# Where:
#   - p = win rate (probability of success)
#   - Weighted_R = sum(allocation_i × R_i) - position-weighted reward-to-risk ratio
#   - R_i = |TP_i - Entry| / |SL - Entry| - individual R-ratio for each TP level
#
# SAFETY CONSTRAINT LAYERS (Applied Sequentially):
# ================================================
# 1. Quarter-Kelly: Raw Kelly × 0.25 (conservative base multiplier)
# 2. Multi-TP Adjustment: Quarter-Kelly × 0.90 (complexity reduction)
# 3. Confidence Scaling: Adjusted Kelly × (confidence_level / 100)
# 4. Kelly Cap: min(Confidence-Scaled, 0.50) - never exceed 50% Kelly
# 5. Hard Risk Cap: min(Kelly-Capped, 0.05) - never risk more than 5% capital
#
# SCENARIO PROBABILITY MODEL:
# ==========================
# Uses distance-based exponential decay with sequential penalty:
# P(scenario) = win_rate × distance_penalty × sequential_penalty
# Where:
#   - distance_penalty = 0.5^(price_distance/0.1) - exponential distance penalty
#   - sequential_penalty = 0.5^(num_tps_in_scenario-1) - sequential execution penalty
# This replaces arbitrary decay factors with mathematically justified probability distribution
#
# PRECISION & SAFETY:
# ==================
# - High-precision decimal arithmetic (28 decimal places) for accurate financial calculations
# - Comprehensive input validation with institutional risk management standards
# - Multiple safety layers prevent over-leveraging and excessive risk-taking
# - Sequential TP scenario logic prevents impossible market outcomes
# - Probability-weighted volatility for accurate risk-adjusted scoring

Risk Management Philosophy:
- Capital preservation is paramount over profit maximization
- Mathematical rigor with 4+ decimal precision in calculations
- Conservative position sizing with multiple safety layers
- Comprehensive validation and error handling
- Scenario-based expected value analysis

Author: Abedalaziz Alzweidi for Zawad Risk Management System
Version: 3.9 - Production-Ready Multi-Level Take Profit System
"""

# CORE IMPORTS:
# =============
import sys          # System operations and exit handling
import math         # Mathematical functions (sqrt, etc.) for Decimal compatibility
from decimal import Decimal, getcontext, ROUND_HALF_UP  # High-precision arithmetic
from typing import List, Dict, Optional, Tuple, NamedTuple  # Type hints for code clarity
from dataclasses import dataclass  # Clean data structure definitions


# PRECISION CONFIGURATION:
# =======================
# Set ultra-high precision for financial calculations (28 decimal places)
# This ensures accurate position sizing and risk calculations even with large capital amounts
getcontext().prec = 28


# ================================================================================================
# CORE DATA STRUCTURES
# ================================================================================================

@dataclass
class TakeProfitLevel:
    """
    Data structure representing a single take profit level with allocation.
    
    This class encapsulates a single TP level in a multi-level strategy, storing both
    the price target and the position percentage to close at that level. The derived
    metrics are calculated during trade setup initialization.
    
    # MATHEMATICAL RELATIONSHIPS:
    # ===========================
    # distance_from_entry = |TP_price - Entry_price|
    # r_ratio = distance_from_entry / |Stop_Loss - Entry_price|
    # 
    # The r_ratio represents the reward-to-risk ratio for this specific TP level,
    # which is later weighted by allocation percentage in the overall strategy.
    
    Attributes:
        price: Take profit price level (Decimal for precision)
        allocation_percentage: Position percentage to close at this level (0-100)
        
    Derived attributes (calculated in __post_init__):
        distance_from_entry: Absolute price distance from entry point
        individual_rr_ratio: Individual reward-to-risk ratio for this TP level
    """
    price: Decimal
    allocation_percentage: Decimal  # Must sum to 100% across all TP levels
    
    # Derived metrics calculated after initialization
    distance_from_entry: Optional[Decimal] = None  # |TP - Entry|
    individual_rr_ratio: Optional[Decimal] = None              # Reward/Risk for this TP
    
    def __post_init__(self):
        """Ensure all values are Decimal type for high-precision calculations."""
        # Convert to Decimal to maintain precision throughout calculations
        self.price = Decimal(str(self.price))
        self.allocation_percentage = Decimal(str(self.allocation_percentage))
        if self.distance_from_entry is not None:
            self.distance_from_entry = Decimal(str(self.distance_from_entry))
        if self.individual_rr_ratio is not None:
            self.individual_rr_ratio = Decimal(str(self.individual_rr_ratio))


@dataclass
class EnhancedTradeSetup:
    """
    Complete trade setup with multi-level take profits.
    
    This is the core data structure that encapsulates an entire trading strategy setup,
    including all TP levels, risk parameters, and trader confidence metrics. The class
    automatically calculates derived metrics upon initialization.
    
    # MATHEMATICAL FOUNDATION:
    # ========================
    # The setup serves as input for the Enhanced Kelly Criterion calculation:
    # 
    # Weighted_R = sum(allocation_i × R_i) where R_i = |TP_i - Entry| / |SL - Entry|
    # Kelly = (win_rate × (Weighted_R + 1) - 1) / Weighted_R
    # 
    # Each TP level contributes to the overall strategy based on its allocation
    # percentage and individual risk-reward ratio.
    
    Attributes:
        capital: Total trading capital available (Decimal for precision)
        entry: Entry price level (must be between SL and all TPs)
        stop_loss: Stop loss price level (defines maximum risk)
        take_profits: List of TP levels with allocations (must sum to 100%)
        direction: Trade direction ('LONG' or 'SHORT') - validates TP positioning
        win_rate: Historical win rate (0.01 to 0.99) - probability of success
        confidence: Confidence level (1 to 100) - trader's conviction in setup
        leverage: Trading leverage (1 to 100) - amplifies both risk and reward
    """
    capital: Decimal         # Total available capital
    entry: Decimal          # Entry price level
    stop_loss: Decimal      # Stop loss price (maximum risk point)
    take_profits: List[TakeProfitLevel]  # List of TP levels with allocations
    direction: str          # 'LONG' or 'SHORT' - validates setup logic
    win_rate: Decimal       # Historical success rate (0.01-0.99)
    confidence: Decimal     # Trader confidence (1-100) - affects final sizing
    leverage: Decimal       # Trading leverage (1-100) - risk amplifier
    
    def __post_init__(self):
        """Ensure all values are Decimal type and calculate derived metrics."""
        # Convert all inputs to high-precision Decimal for accurate calculations
        self.capital = Decimal(str(self.capital))
        self.entry = Decimal(str(self.entry))
        self.stop_loss = Decimal(str(self.stop_loss))
        self.win_rate = Decimal(str(self.win_rate))
        self.confidence = Decimal(str(self.confidence))
        self.leverage = Decimal(str(self.leverage))
        
        # Calculate derived metrics for all TP levels
        # This populates distance_from_entry and individual_rr_ratio for each TP
        self._calculate_tp_metrics()
    
    def _calculate_tp_metrics(self):
        """Calculate distance from entry and individual R-ratios for each TP level.
        
        # MATHEMATICAL FORMULAS:
        # =====================
        # distance_from_entry = |TP_price - Entry_price|
        # individual_rr_ratio = distance_from_entry / |Stop_Loss - Entry_price|
        
        This method populates the derived attributes for each TP level, which are later
        used in weighted R-ratio calculation and scenario probability modeling.
        """
        # Calculate the risk distance (entry to stop loss)
        sl_distance = abs(self.entry - self.stop_loss)
        
        # Calculate metrics for each TP level
        for tp in self.take_profits:
            # Distance from entry to this TP level
            tp.distance_from_entry = abs(tp.price - self.entry)
            
            # Individual R-ratio: reward distance / risk distance
            if sl_distance > 0:
                tp.individual_rr_ratio = tp.distance_from_entry / sl_distance
            else:
                # Edge case: if SL distance is zero, set R-ratio to zero
                tp.individual_rr_ratio = Decimal('0')


class ValidationResult(NamedTuple):
    """Result of input validation with success status and any error messages.
    
    # PURPOSE:
    # ========
    # This simple data structure encapsulates validation results, allowing
    # the validation methods to return both success/failure status and detailed
    # error messages for debugging and user feedback.
    """
    is_valid: bool          # True if all validation checks passed
    error_message: str = "" # Detailed error description if validation failed


class RiskMetrics(NamedTuple):
    """Comprehensive risk assessment metrics for trading strategy evaluation.
    
    # MATHEMATICAL FOUNDATION:
    # =======================
    # These metrics provide a holistic view of the trading strategy's risk profile:
    # 
    # expected_value = sum(probability_i × return_i) - probability-weighted expected return
    # risk_adjusted_score = expected_value / volatility - Sharpe-like ratio
    # probability_of_profit = sum(probability_i) for all profitable scenarios
    # maximum_drawdown_impact = worst_case_loss × final_risk_fraction - capital impact
    # 
    # Breakeven calculations help traders understand minimum performance requirements.
    """
    expected_value: Decimal           # Probability-weighted expected return (%)
    worst_case_loss: Decimal         # Maximum potential loss scenario (%)
    best_case_return: Decimal        # Maximum potential gain scenario (%)
    probability_of_profit: Decimal   # Combined probability of all profitable scenarios
    risk_adjusted_score: Decimal     # Expected return / volatility (Sharpe-like ratio)
    maximum_drawdown_impact: Decimal # Worst-case impact on total capital (%)
    win_rate_breakeven: Decimal      # Minimum win rate needed to breakeven
    r_ratio_breakeven: Decimal       # Minimum R-ratio needed to breakeven


class PositionSizingResult(NamedTuple):
    """Complete position sizing calculation results with all safety constraint layers.
    
    # SAFETY CONSTRAINT LAYERS:
    # ========================
    # This structure captures each step of the multi-layered safety system:
    # 
    # 1. kelly_raw: Initial Kelly Criterion calculation
    # 2. kelly_quarter: Raw Kelly × 0.25 (Quarter-Kelly conservative base)
    # 3. kelly_multi_tp_adjusted: Quarter-Kelly × 0.90 (Multi-TP complexity reduction)
    # 4. kelly_confidence_scaled: Adjusted Kelly × (confidence / 100)
    # 5. kelly_capped: min(Confidence-Scaled, 0.50) - 50% Kelly maximum
    # 6. final_risk_fraction: min(Kelly-Capped, 0.05) - 5% capital maximum
    # 
    # Each layer provides additional safety, ensuring conservative position sizing.
    """
    kelly_raw: Decimal                # Initial Kelly Criterion calculation
    kelly_quarter: Decimal           # Raw Kelly × 0.25 (conservative base)
    kelly_multi_tp_adjusted: Decimal # Quarter-Kelly × 0.90 (complexity adjustment)
    kelly_confidence_scaled: Decimal # Adjusted Kelly × (confidence / 100)
    kelly_capped: Decimal           # min(Confidence-Scaled, 0.50) - Kelly cap
    final_risk_fraction: Decimal    # min(Kelly-Capped, 0.05) - final risk %
    capital_at_risk: Decimal        # Dollar amount at risk (final_risk × capital)
    base_position_size: Decimal     # Position size without leverage
    leveraged_position_size: Decimal # Position size with leverage applied
    notional_exposure: Decimal      # Total market exposure (leveraged position)


# ================================================================================================
# ZAWAD RISK ENGINE v3.9 - MAIN CLASS
# ================================================================================================

class ZawadRiskEngine:
    """
    Advanced risk management engine implementing Enhanced Kelly Criterion with multi-level
    take profit position sizing and comprehensive scenario analysis.
    
    # CORE CAPABILITIES:
    # ==================
    # 1. Enhanced Kelly Criterion with weighted R-ratios for multi-TP strategies
    # 2. Multi-layered safety constraints (5 sequential safety layers)
    # 3. Distance-based scenario probability modeling (replaced arbitrary decay)
    # 4. Sequential TP scenario generation (eliminates impossible combinations)
    # 5. Comprehensive risk metrics with probability-weighted volatility
    # 6. Leverage-adjusted Kelly sizing (prevents systematic over-leveraging)
    # 7. High-precision decimal arithmetic for financial accuracy
    # 8. Production-ready validation and error handling
    
    # MATHEMATICAL FRAMEWORK:
    # ======================
    # Enhanced Kelly: k = (p × (Weighted_R + 1) - 1) / Weighted_R
    # Weighted R-Ratio: sum(allocation_i × R_i)
    # Safety Layers: Quarter-Kelly -> Multi-TP -> Confidence -> Kelly Cap -> Hard Cap
    # Scenario Probabilities: win_rate × distance_penalty × sequential_penalty
    # Risk-Adjusted Score: expected_return / probability_weighted_volatility
    
    This class handles complex position sizing calculations for leveraged trading with emphasis on
    mathematical accuracy, capital preservation, and comprehensive risk assessment.
    """
    
    def __init__(self):
        """Initialize the risk engine with conservative safety parameters.
        
        # SAFETY PHILOSOPHY:
        # ==================
        # The initialization sets up multiple layers of conservative constraints to ensure
        # that position sizing never exceeds safe risk levels, even with aggressive inputs.
        # Each parameter has been carefully chosen based on institutional risk management
        # best practices and extensive backtesting.
        """
        
        # ========================================================================================
        # CORE SAFETY CONSTRAINT PARAMETERS
        # ========================================================================================
        
        # Layer 1: Quarter-Kelly Base (Conservative Foundation)
        self.SAFETY_MULTIPLIER = Decimal('0.25')         # 25% of raw Kelly (industry standard)
        
        # Layer 2: Multi-TP Complexity Adjustment
        self.MULTI_TP_ADJUSTMENT = Decimal('0.90')       # 10% reduction for execution complexity
        
        # Layer 3: Kelly Absolute Cap
        self.KELLY_CAP = Decimal('0.50')                 # Never exceed 50% Kelly (mathematical limit)
        
        # Layer 4: Hard Risk Cap (Ultimate Safety)
        self.HARD_RISK_CAP = Decimal('0.05')             # Never risk >5% of capital per trade
        
        # Strategy Quality Threshold
        self.MIN_RR_RATIO = Decimal('1.0')               # Minimum 1:1 reward-to-risk ratio
        
        # ========================================================================================
        # INPUT VALIDATION BOUNDARIES
        # ========================================================================================
        
        # Win Rate Boundaries (Prevents Unrealistic Inputs)
        self.MIN_WIN_RATE = Decimal('0.01')              # Minimum 1% (prevents zero division)
        self.MAX_WIN_RATE = Decimal('0.99')              # Maximum 99% (prevents overconfidence)
        
        # Confidence Level Boundaries
        self.MIN_CONFIDENCE = Decimal('1')               # Minimum 1% confidence
        self.MAX_CONFIDENCE = Decimal('100')             # Maximum 100% confidence
        
        # Leverage Boundaries (Risk Management)
        self.MIN_LEVERAGE = Decimal('1')                 # Minimum 1x (no leverage)
        self.MAX_LEVERAGE = Decimal('100')               # Maximum 100x (extreme but allowed)
        
        # ========================================================================================
        # SCENARIO PROBABILITY MODEL PARAMETERS (NEW DISTANCE-BASED MODEL)
        # ========================================================================================
        
        # These parameters control the new mathematically-sound probability model
        # that replaced the arbitrary 0.8 decay factor:
        #
        # distance_penalty = 0.5^(price_distance/0.1)
        # sequential_penalty = 0.5^(num_tps_in_scenario-1)
        # P(scenario) = win_rate × distance_penalty × sequential_penalty
    
    # ============================================================================================
    # VALIDATION METHODS
    # ============================================================================================
    
    def _validate_basic_params(self, setup: EnhancedTradeSetup) -> Optional[ValidationResult]:
        """Validate basic parameter ranges and types."""
        if setup.capital <= 0:
            return ValidationResult(False, "Capital must be positive")
        
        if setup.entry <= 0:
            return ValidationResult(False, "Entry price must be positive")
        
        if setup.stop_loss <= 0:
            return ValidationResult(False, "Stop loss price must be positive")
        
        if setup.win_rate < self.MIN_WIN_RATE or setup.win_rate > self.MAX_WIN_RATE:
            return ValidationResult(False, f"Win rate must be between {self.MIN_WIN_RATE} and {self.MAX_WIN_RATE}")
        
        if setup.confidence < self.MIN_CONFIDENCE or setup.confidence > self.MAX_CONFIDENCE:
            return ValidationResult(False, f"Confidence must be between {self.MIN_CONFIDENCE} and {self.MAX_CONFIDENCE}")
        
        if setup.leverage < self.MIN_LEVERAGE or setup.leverage > self.MAX_LEVERAGE:
            return ValidationResult(False, f"Leverage must be between {self.MIN_LEVERAGE} and {self.MAX_LEVERAGE}")
        
        return None
    
    def _validate_direction_logic(self, setup: EnhancedTradeSetup) -> Optional[ValidationResult]:
        """Validate trade direction and stop loss positioning."""
        direction = setup.direction.upper()
        if direction not in ['LONG', 'SHORT']:
            return ValidationResult(False, "Direction must be 'LONG' or 'SHORT'")
        
        # Validate stop loss positioning
        if direction == 'LONG' and setup.stop_loss >= setup.entry:
            return ValidationResult(False, "For LONG trades, stop loss must be below entry price")
        
        if direction == 'SHORT' and setup.stop_loss <= setup.entry:
            return ValidationResult(False, "For SHORT trades, stop loss must be above entry price")
        
        return None
    
    def _validate_tp_prices_and_allocations(self, setup: EnhancedTradeSetup) -> Optional[ValidationResult]:
        """Validate TP prices and allocation percentages are positive."""
        for i, tp in enumerate(setup.take_profits, 1):
            if tp.price <= 0:
                return ValidationResult(False, f"TP{i} price must be positive")
            
            if tp.allocation_percentage <= 0:
                return ValidationResult(False, f"TP{i} allocation must be positive")
        
        return None
    
    def _validate_long_tp_sequence(self, setup: EnhancedTradeSetup, tp_prices: List[Decimal]) -> Optional[ValidationResult]:
        """Validate LONG trade TP sequence logic (above entry, ascending order)."""
        # For LONG: all TPs must be above entry and in ascending order
        for i, tp_price in enumerate(tp_prices, 1):
            if tp_price <= setup.entry:
                return ValidationResult(False, f"For LONG trades, TP{i} must be above entry price")
        
        if tp_prices != sorted(tp_prices):
            return ValidationResult(False, "For LONG trades, TP levels must be in ascending order")
        
        return None
    
    def _validate_short_tp_sequence(self, setup: EnhancedTradeSetup, tp_prices: List[Decimal]) -> Optional[ValidationResult]:
        """Validate SHORT trade TP sequence logic (below entry, descending order)."""
        # For SHORT: all TPs must be below entry and in descending order
        for i, tp_price in enumerate(tp_prices, 1):
            if tp_price >= setup.entry:
                return ValidationResult(False, f"For SHORT trades, TP{i} must be below entry price")
        
        if tp_prices != sorted(tp_prices, reverse=True):
            return ValidationResult(False, "For SHORT trades, TP levels must be in descending order")
        
        return None
    
    def _validate_take_profits(self, setup: EnhancedTradeSetup) -> Optional[ValidationResult]:
        """Validate take profit levels, prices, and sequence logic."""
        if not setup.take_profits or len(setup.take_profits) == 0:
            return ValidationResult(False, "At least one take profit level is required")
        
        if len(setup.take_profits) > 4:
            return ValidationResult(False, "Maximum 4 take profit levels allowed")
        
        # Validate TP prices and allocations are positive
        err = self._validate_tp_prices_and_allocations(setup)
        if err:
            return err
        
        # Validate TP sequence logic based on direction
        direction = setup.direction.upper()
        tp_prices = [tp.price for tp in setup.take_profits]
        
        if direction == 'LONG':
            err = self._validate_long_tp_sequence(setup, tp_prices)
        else:  # SHORT
            err = self._validate_short_tp_sequence(setup, tp_prices)
        
        if err:
            return err
        
        return None
    
    def _validate_allocation_sum(self, setup: EnhancedTradeSetup) -> Optional[ValidationResult]:
        """Validate that TP allocations sum to exactly 100%."""
        total_allocation = sum(tp.allocation_percentage for tp in setup.take_profits)
        if abs(total_allocation - Decimal('100')) > Decimal('0.01'):
            return ValidationResult(False, f"TP allocations must sum to exactly 100%, got {total_allocation}%")
        
        return None
    
    def _validate_rr_ratios(self, setup: EnhancedTradeSetup) -> Optional[ValidationResult]:
        """Validate individual R-ratio minimums for each TP level."""
        for i, tp in enumerate(setup.take_profits, 1):
            if tp.individual_rr_ratio < self.MIN_RR_RATIO:
                return ValidationResult(False, f"TP{i} R-ratio ({tp.individual_rr_ratio:.4f}) below minimum ({self.MIN_RR_RATIO})")
        
        return None
    
    def validate_enhanced_inputs(self, setup: EnhancedTradeSetup) -> ValidationResult:
        """
        Comprehensive validation of enhanced trade setup inputs.
        
        Validates all inputs according to institutional risk management standards:
        - Basic parameter ranges and types
        - Trade direction consistency (TP levels align with LONG/SHORT)
        - Stop loss positioning relative to entry
        - TP allocation sum verification (must equal 100%)
        - Individual R-ratio minimums
        - Economic sense validation
        
        Args:
            setup: Enhanced trade setup to validate
            
        Returns:
            ValidationResult with success status and error message if invalid
        """
        try:
            # Phase 1: Basic Parameter Validation
            err = self._validate_basic_params(setup)
            if err:
                return err
            
            # Phase 2: Direction and Trade Logic Validation
            err = self._validate_direction_logic(setup)
            if err:
                return err
            
            # Phase 3: Take Profit Validation
            err = self._validate_take_profits(setup)
            if err:
                return err
            
            # Phase 4: Allocation Sum Validation
            err = self._validate_allocation_sum(setup)
            if err:
                return err
            
            # Phase 5: R-Ratio Validation
            err = self._validate_rr_ratios(setup)
            if err:
                return err
            
            return ValidationResult(True, "")
            
        except Exception as e:
            return ValidationResult(False, f"Validation error: {str(e)}")
    
    # ============================================================================================
    # MATHEMATICAL ENGINE
    # ============================================================================================
    
    def calculate_weighted_rr_ratio(self, setup: EnhancedTradeSetup) -> Decimal:
        """
        Calculate position-weighted R-ratio for multi-level take profit strategy.
        
        # MATHEMATICAL FOUNDATION:
        # =======================
        # The weighted R-ratio represents the expected reward-to-risk ratio when considering
        # the allocation percentages across all TP levels. This is crucial for accurate
        # Kelly Criterion calculation in multi-TP strategies.
        #
        # Formula: Weighted_R = sum(allocation_i × R_i)
        # Where:
        #   - allocation_i = percentage of position closed at TP_i (as decimal fraction)
        #   - R_i = |TP_i - Entry| / |SL - Entry| (individual R-ratio for TP_i)
        #
        # EXAMPLE:
        # ========
        # TP1: 50% allocation, 2:1 R-ratio -> contributes 0.50 × 2.0 = 1.0
        # TP2: 30% allocation, 4:1 R-ratio -> contributes 0.30 × 4.0 = 1.2  
        # TP3: 20% allocation, 6:1 R-ratio -> contributes 0.20 × 6.0 = 1.2
        # Weighted_R = 1.0 + 1.2 + 1.2 = 3.4
        
        Args:
            setup: Enhanced trade setup with validated TP levels
            
        Returns:
            Weighted R-ratio as high-precision Decimal
        """
        weighted_r = Decimal('0')  # Initialize weighted R-ratio accumulator
        
        # Calculate contribution from each TP level
        for tp in setup.take_profits:
            # Convert allocation percentage to decimal fraction (e.g., 25% -> 0.25)
            allocation_fraction = tp.allocation_percentage / Decimal('100')
            
            # Add this TP's weighted contribution to the total
            # weighted_contribution = allocation_fraction × individual_R_ratio
            weighted_r += allocation_fraction * tp.individual_rr_ratio
        
        return weighted_r
    
    def calculate_enhanced_kelly(self, setup: EnhancedTradeSetup) -> PositionSizingResult:
        """
        Calculate Enhanced Kelly Criterion with multi-layered safety constraints.
        
        # ENHANCED KELLY CRITERION FRAMEWORK:
        # ==================================
        # This method implements a sophisticated, multi-layered approach to position sizing
        # that builds upon the classic Kelly Criterion with additional safety constraints
        # specifically designed for leveraged trading environments.
        #
        # BASE KELLY FORMULA: k = (p × (R + 1) - 1) / R
        # Where: p = win_rate, R = reward-to-risk ratio, k = Kelly fraction
        #
        # ENHANCED KELLY FORMULA: k = (p × (Weighted_R + 1) - 1) / Weighted_R
        # Where: Weighted_R = sum(allocation_i × R_i) for multi-TP strategies
        
        # FIVE-LAYER SAFETY CONSTRAINT SYSTEM:
        # ====================================
        # Each layer applies progressively more conservative constraints to ensure
        # position sizing never exceeds safe risk levels, even with aggressive inputs.
        #
        # Layer 1: QUARTER-KELLY BASE (Conservative Foundation)
        #   - Raw Kelly × 0.25 (industry standard for institutional trading)
        #   - Protects against Kelly's theoretical assumptions being violated
        #
        # Layer 2: MULTI-TP COMPLEXITY ADJUSTMENT
        #   - Quarter-Kelly × 0.90 (10% reduction for execution complexity)
        #   - Accounts for slippage and execution risk in multi-level strategies
        #
        # Layer 3: CONFIDENCE SCALING
        #   - Adjusted Kelly × (confidence_level / 100)
        #   - Allows traders to scale down based on setup confidence
        #
        # Layer 4: KELLY ABSOLUTE CAP
        #   - min(Confidence-Scaled, 0.50) - never exceed 50% Kelly
        #   - Mathematical boundary to prevent extreme position sizes
        #
        # Layer 5: HARD RISK CAP (Ultimate Safety)
        #   - min(Kelly-Capped, 0.05) - never risk >5% of capital per trade
        #   - Final safety net regardless of all other calculations
        
        Args:
            setup: Validated enhanced trade setup
            
        Returns:
            PositionSizingResult with all calculation steps and final sizing
        """
        # Calculate weighted R-ratio
        weighted_r = self.calculate_weighted_rr_ratio(setup)
        
        # Prevent division by zero
        if weighted_r <= 0:
            raise ValueError(f"Weighted R-ratio must be positive, got {weighted_r}")
        
        # Step 1: Calculate Raw Kelly using Enhanced formula
        # Kelly = (p × (Weighted_R + 1) - 1) / Weighted_R
        kelly_raw = (setup.win_rate * (weighted_r + Decimal('1')) - Decimal('1')) / weighted_r
        
        # Ensure Kelly is non-negative (negative Kelly means unfavorable odds)
        if kelly_raw < 0:
            kelly_raw = Decimal('0')
        
        # Step 1.5: Apply leverage adjustment to Kelly
        # Leverage amplifies volatility, so reduce Kelly fraction proportionally
        # Use sqrt(leverage) since volatility scales with square root of leverage
        leverage_adjustment_factor = Decimal(str(math.sqrt(float(setup.leverage)))) if setup.leverage > 1 else Decimal('1')
        kelly_leverage_adjusted = kelly_raw / leverage_adjustment_factor
        
        # Step 2: Apply Quarter-Kelly safety multiplier
        kelly_quarter = kelly_leverage_adjusted * self.SAFETY_MULTIPLIER
        
        # Step 3: Apply multi-TP complexity adjustment
        kelly_multi_tp_adjusted = kelly_quarter * self.MULTI_TP_ADJUSTMENT
        
        # Step 4: Apply confidence scaling
        confidence_fraction = setup.confidence / Decimal('100')
        kelly_confidence_scaled = kelly_multi_tp_adjusted * confidence_fraction
        
        # Step 5: Apply Kelly hard cap (never exceed 50% Kelly)
        kelly_capped = min(kelly_confidence_scaled, self.KELLY_CAP)
        
        # Step 6: Apply absolute risk hard cap (never risk more than 5% of capital)
        final_risk_fraction = min(kelly_capped, self.HARD_RISK_CAP)
        
        # Calculate position sizing values
        capital_at_risk = setup.capital * final_risk_fraction
        
        # Calculate base position size (risk amount / stop loss distance as percentage)
        sl_distance_pct = abs(setup.entry - setup.stop_loss) / setup.entry
        base_position_size = capital_at_risk / sl_distance_pct
        
        # Calculate leveraged position size
        leveraged_position_size = base_position_size * setup.leverage
        
        # Calculate notional exposure
        notional_exposure = leveraged_position_size
        
        return PositionSizingResult(
            kelly_raw=kelly_raw,
            kelly_quarter=kelly_quarter,
            kelly_multi_tp_adjusted=kelly_multi_tp_adjusted,
            kelly_confidence_scaled=kelly_confidence_scaled,
            kelly_capped=kelly_capped,
            final_risk_fraction=final_risk_fraction,
            capital_at_risk=capital_at_risk,
            base_position_size=base_position_size,
            leveraged_position_size=leveraged_position_size,
            notional_exposure=notional_exposure
        )
    


    def _generate_sequential_tp_scenarios(self, setup: EnhancedTradeSetup, scenarios: Dict[str, Dict]) -> List[Decimal]:
        """Generate sequential TP scenarios with distance-based probability model."""
        raw_probabilities = []
        num_tps = len(setup.take_profits)
        
        for i in range(1, num_tps + 1):
            # Sequential scenario: TP1, TP1_TP2, TP1_TP2_TP3, etc.
            tp_indices = list(range(i))  # [0], [0,1], [0,1,2], etc.
            scenario_name = 'TP' + '_TP'.join(str(idx + 1) for idx in tp_indices)
            
            # Calculate sequential return (sum of all TPs hit up to this point)
            total_return = Decimal('0')
            for tp_idx in tp_indices:
                tp = setup.take_profits[tp_idx]
                allocation_fraction = tp.allocation_percentage / Decimal('100')
                tp_return = allocation_fraction * tp.individual_rr_ratio
                total_return += tp_return
            
            # Convert to percentage
            return_percentage = total_return * Decimal('100')
            
            # Calculate mathematically correct sequential probability
            scenario_prob = self._calculate_tp_scenario_probability(setup, i)
            raw_probabilities.append(scenario_prob)
            
            scenarios[scenario_name] = {
                'probability': scenario_prob,  # Will be normalized later
                'outcome': scenario_name,
                'return_percentage': return_percentage,
                'description': f'Sequential TPs 1-{i} hit, progression stops at TP{i}'
            }
        
        return raw_probabilities
    
    def _calculate_tp_scenario_probability(self, setup: EnhancedTradeSetup, tp_level: int) -> Decimal:
        """Calculate probability for reaching specific TP level using distance-based model."""
        # Base probability from win rate
        base_prob = setup.win_rate
        
        # Distance penalty: exponential decay based on cumulative distance to final TP
        final_tp = setup.take_profits[tp_level - 1]  # Final TP in this scenario
        cumulative_distance = abs(final_tp.price - setup.entry) / setup.entry
        
        # Strong distance penalty: P = 0.5^(distance/0.1) ensures distant TPs are penalized heavily
        distance_penalty = Decimal('0.5') ** (cumulative_distance / Decimal('0.1'))
        
        # Sequential penalty: GUARANTEE monotonic decrease
        sequential_penalty = Decimal('0.5') ** (tp_level - 1)
        
        # Combine all factors
        return base_prob * distance_penalty * sequential_penalty
    
    def _normalize_scenario_probabilities(self, scenarios: Dict[str, Dict], raw_probabilities: List[Decimal], setup: EnhancedTradeSetup) -> None:
        """Normalize scenario probabilities to sum to 1.0 with proper fallback handling."""
        # Normalize profit scenario probabilities to sum to win_rate
        total_profit_prob = sum(raw_probabilities)
        if total_profit_prob > 0:
            normalization_factor = setup.win_rate / total_profit_prob
            for scenario_name, scenario_prob in zip(scenarios.keys(), raw_probabilities):
                if scenario_name != 'WORST_CASE':
                    scenarios[scenario_name]['probability'] = scenario_prob * normalization_factor
        
        # Final normalization to ensure sum = 1.0
        total_prob = sum(scenario['probability'] for scenario in scenarios.values())
        if total_prob > 0 and total_prob != Decimal('1'):
            # Standard normalization when total_prob is valid
            for scenario in scenarios.values():
                scenario['probability'] = scenario['probability'] / total_prob
        elif total_prob == 0:
            # Fallback: equal probability distribution when all probabilities are zero
            uniform_prob = Decimal('1') / len(scenarios)
            for scenario in scenarios.values():
                scenario['probability'] = uniform_prob
            print("WARNING: All scenario probabilities were zero, using uniform distribution")
        
        # Verify normalization worked
        final_total = sum(scenario['probability'] for scenario in scenarios.values())
        if abs(final_total - Decimal('1')) > Decimal('0.001'):
            print(f"WARNING: Probabilities sum to {final_total}, not 1.0")
    
    def calculate_scenario_matrix(self, setup: EnhancedTradeSetup) -> Dict[str, Dict]:
        """
        Generate realistic sequential scenario matrix for take profit execution.
        
        # MAJOR MATHEMATICAL IMPROVEMENTS IMPLEMENTED:
        # ===========================================
        # 1. SEQUENTIAL LOGIC: Replaced impossible combinatorial scenarios (TP2 only, TP1_TP3) 
        #    with realistic sequential scenarios (TP1, TP1_TP2, TP1_TP2_TP3)
        # 2. DISTANCE-BASED PROBABILITIES: Replaced arbitrary 0.8 decay with mathematically 
        #    sound distance-based exponential penalty
        # 3. EXPONENTIAL DISTANCE PENALTY: P = 0.5^(distance/0.1) heavily penalizes distant TPs
        # 4. SEQUENTIAL PENALTY: P = 0.5^(num_tps-1) ensures monotonic probability decrease
        # 5. PROBABILITY NORMALIZATION: Guarantees probabilities sum to exactly 1.0
        
        # SEQUENTIAL TP SCENARIOS (Only Realistic Outcomes):
        # =================================================
        # - WORST_CASE: Only SL hit (-100% of risk)
        # - TP1: Only TP1 hit (sequential progression stops at TP1)
        # - TP1_TP2: TP1 and TP2 hit (sequential progression stops at TP2)
        # - TP1_TP2_TP3: TP1, TP2, and TP3 hit (sequential progression stops at TP3)
        # - etc.
        #
        # This eliminates impossible scenarios like "TP2 only" or "TP1_TP3" which
        # violate sequential price movement logic in financial markets.
        
        # MATHEMATICAL FORMULA:
        # ====================
        # P(scenario_i) = win_rate × distance_penalty × sequential_penalty
        # Where:
        #   - distance_penalty = 0.5^(price_distance/0.1) [exponential distance decay]
        #   - sequential_penalty = 0.5^(i-1) [sequential execution difficulty]
        #   - Normalization ensures sum(P(all_scenarios)) = 1.0
        
        Args:
            setup: Validated enhanced trade setup
            
        Returns:
            Dictionary of scenarios with probabilities and returns
        """
        scenarios = {}
        
        # Calculate base probabilities
        prob_sl = Decimal('1') - setup.win_rate  # Probability of SL hit
        
        # Scenario 1: WORST_CASE - Only SL hit
        scenarios['WORST_CASE'] = {
            'probability': prob_sl,
            'outcome': 'SL_HIT',
            'return_percentage': Decimal('-100'),
            'description': 'Stop loss hit, no TPs reached'
        }
        
        # Generate sequential TP scenarios using helper method
        raw_probabilities = self._generate_sequential_tp_scenarios(setup, scenarios)
        
        # Normalize probabilities using helper method
        self._normalize_scenario_probabilities(scenarios, raw_probabilities, setup)
        
        return scenarios
    
    def calculate_risk_metrics(self, setup: EnhancedTradeSetup, position_result: PositionSizingResult) -> RiskMetrics:
        """
        Calculate comprehensive risk assessment metrics.
        
        Args:
            setup: Validated enhanced trade setup
            position_result: Position sizing calculation results
            
        Returns:
            RiskMetrics with comprehensive risk assessment
        """
        scenarios = self.calculate_scenario_matrix(setup)
        
        # Calculate expected value
        expected_return_pct = Decimal('0')
        for scenario in scenarios.values():
            expected_return_pct += scenario['probability'] * scenario['return_percentage']
        
        expected_value = position_result.capital_at_risk * expected_return_pct / Decimal('100')
        
        # Calculate best and worst case outcomes
        worst_case_loss = position_result.capital_at_risk  # Maximum loss is the capital at risk
        
        best_case_return = Decimal('0')
        for scenario in scenarios.values():
            if scenario['return_percentage'] > 0:
                scenario_return = position_result.capital_at_risk * scenario['return_percentage'] / Decimal('100')
                if scenario_return > best_case_return:
                    best_case_return = scenario_return
        
        # Calculate probability of profit (sum of all profitable scenarios)
        probability_of_profit = Decimal('0')
        for scenario in scenarios.values():
            if scenario['return_percentage'] > 0:
                probability_of_profit += scenario['probability']
        
        # Calculate risk-adjusted score (Sharpe-like ratio)
        # Risk-adjusted score = Expected Return / Position Volatility
        # Use probability-weighted volatility for mathematical consistency
        
        # Calculate probability-weighted variance: sum(p_i × (r_i - E[r])²)
        weighted_variance = Decimal('0')
        for scenario in scenarios.values():
            deviation = scenario['return_percentage'] - expected_return_pct
            weighted_variance += scenario['probability'] * (deviation ** 2)
        
        # Calculate volatility with appropriate fallback for zero variance
        volatility = Decimal(str(math.sqrt(float(weighted_variance)))) if weighted_variance > 0 else Decimal('0.01')  # Small epsilon instead of 1
        risk_adjusted_score = expected_return_pct / volatility if volatility > 0 else Decimal('0')
        
        # Calculate maximum drawdown impact
        maximum_drawdown_impact = (worst_case_loss / setup.capital) * Decimal('100')
        
        # Calculate breakeven metrics
        weighted_r = self.calculate_weighted_rr_ratio(setup)
        win_rate_breakeven = Decimal('1') / (weighted_r + Decimal('1'))
        r_ratio_breakeven = (Decimal('1') - setup.win_rate) / setup.win_rate
        
        return RiskMetrics(
            expected_value=expected_value,
            worst_case_loss=worst_case_loss,
            best_case_return=best_case_return,
            probability_of_profit=probability_of_profit,
            risk_adjusted_score=risk_adjusted_score,
            maximum_drawdown_impact=maximum_drawdown_impact,
            win_rate_breakeven=win_rate_breakeven,
            r_ratio_breakeven=r_ratio_breakeven
        )
    
    # ============================================================================================
    # CLI INTERFACE METHODS
    # ============================================================================================
    
    def get_enhanced_trade_setup(self) -> EnhancedTradeSetup:
        """
        Interactive CLI input collection for enhanced trade setup.
        
        Collects all required parameters with validation and provides preset options
        for TP allocation strategies.
        
        Returns:
            EnhancedTradeSetup with all validated inputs
        """
        print("\n" + "=" * 80)
        print("ZAWAD RISK ENGINE v3.9 - Multi-Level Take Profit System")
        print("=" * 80)
        print("\nSTEP 1: Base Trade Setup")
        print("-" * 25)
        
        # Collect basic parameters
        capital = self._get_decimal_input("Trading Capital ($): ", min_value=Decimal('1'))
        entry = self._get_decimal_input("Entry Price: ", min_value=Decimal('0.001'))
        stop_loss = self._get_decimal_input("Stop Loss Price: ", min_value=Decimal('0.001'))
        win_rate = self._get_decimal_input("Historical Win Rate (0.01-0.99): ", 
                                         min_value=Decimal('0.01'), max_value=Decimal('0.99'))
        confidence = self._get_decimal_input("Confidence Level (1-100): ", 
                                           min_value=Decimal('1'), max_value=Decimal('100'))
        leverage = self._get_decimal_input("Leverage (1-100): ", 
                                         min_value=Decimal('1'), max_value=Decimal('100'))
        
        # Determine trade direction
        direction = self._get_direction_input(entry, stop_loss)
        
        print("\nSTEP 2: Take Profit Configuration")
        print("-" * 32)
        
        # Get number of TP levels
        num_tps = self._get_integer_input("Number of TP Levels (1-4): ", min_value=1, max_value=4)
        
        # Get TP levels with prices and allocations
        take_profits = self._get_take_profit_levels(num_tps, entry, direction)
        
        return EnhancedTradeSetup(
            capital=capital,
            entry=entry,
            stop_loss=stop_loss,
            take_profits=take_profits,
            direction=direction,
            win_rate=win_rate,
            confidence=confidence,
            leverage=leverage
        )
    
    def _get_decimal_input(self, prompt: str, min_value: Decimal = None, max_value: Decimal = None) -> Decimal:
        """Get validated decimal input from user."""
        while True:
            try:
                value = Decimal(input(prompt).strip())
                if min_value is not None and value < min_value:
                    print(f"Error: Value must be at least {min_value}")
                    continue
                if max_value is not None and value > max_value:
                    print(f"Error: Value must be at most {max_value}")
                    continue
                return value
            except Exception:
                print("Error: Please enter a valid number")
    
    def _get_integer_input(self, prompt: str, min_value: int = None, max_value: int = None) -> int:
        """Get validated integer input from user."""
        while True:
            try:
                value = int(input(prompt).strip())
                if min_value is not None and value < min_value:
                    print(f"Error: Value must be at least {min_value}")
                    continue
                if max_value is not None and value > max_value:
                    print(f"Error: Value must be at most {max_value}")
                    continue
                return value
            except Exception:
                print("Error: Please enter a valid integer")
    
    def _get_direction_input(self, entry: Decimal, stop_loss: Decimal) -> str:
        """Determine trade direction based on entry and stop loss relationship."""
        if stop_loss < entry:
            direction = "LONG"
            print(f"\n✓ Trade Direction: {direction} (SL below entry)")
        elif stop_loss > entry:
            direction = "SHORT" 
            print(f"\n✓ Trade Direction: {direction} (SL above entry)")
        else:
            print("\nError: Stop loss cannot equal entry price")
            sys.exit(1)
        
        return direction
    
    def _get_single_tp_price(self, tp_index: int, entry: Decimal, direction: str, existing_tps: List[TakeProfitLevel]) -> Decimal:
        """Get and validate a single TP price with direction and sequence checks."""
        while True:
            tp_price = self._get_decimal_input(f"TP{tp_index+1} Price: ", min_value=Decimal('0.001'))
            
            # Validate TP direction consistency
            if direction == "LONG" and tp_price <= entry:
                print(f"Error: For LONG trades, TP{tp_index+1} must be above entry price ({entry})")
                continue
            elif direction == "SHORT" and tp_price >= entry:
                print(f"Error: For SHORT trades, TP{tp_index+1} must be below entry price ({entry})")
                continue
            
            # Check sequence order for multiple TPs
            if tp_index > 0:
                prev_tp = existing_tps[tp_index-1].price
                if direction == "LONG" and tp_price <= prev_tp:
                    print(f"Error: TP{tp_index+1} must be higher than TP{tp_index} ({prev_tp}) for LONG trades")
                    continue
                elif direction == "SHORT" and tp_price >= prev_tp:
                    print(f"Error: TP{tp_index+1} must be lower than TP{tp_index} ({prev_tp}) for SHORT trades")
                    continue
            
            return tp_price
    
    def _apply_preset_allocations(self, take_profits: List[TakeProfitLevel], preset_choice: int, num_tps: int) -> None:
        """Apply allocation percentages based on preset choice."""
        if preset_choice == 1:  # Equal Split
            equal_allocation = Decimal('100') / Decimal(str(num_tps))
            for tp in take_profits:
                tp.allocation_percentage = equal_allocation
        
        elif preset_choice == 2:  # Conservative
            conservative_allocations = [Decimal('50'), Decimal('30'), Decimal('15'), Decimal('5')]
            for i, tp in enumerate(take_profits):
                tp.allocation_percentage = conservative_allocations[i] if i < len(conservative_allocations) else Decimal('5')
        
        elif preset_choice == 3:  # Aggressive
            aggressive_allocations = [Decimal('20'), Decimal('30'), Decimal('30'), Decimal('20')]
            for i, tp in enumerate(take_profits):
                tp.allocation_percentage = aggressive_allocations[i] if i < len(aggressive_allocations) else Decimal('20')
    
    def _get_custom_allocations(self, take_profits: List[TakeProfitLevel]) -> None:
        """Get custom allocation percentages from user input."""
        print("\nEnter allocation percentages (must sum to 100%):")
        total_allocation = Decimal('0')
        
        for i, tp in enumerate(take_profits):
            if i == len(take_profits) - 1:  # Last TP gets remainder
                remaining = Decimal('100') - total_allocation
                tp.allocation_percentage = remaining
                print(f"TP{i+1} Allocation: {remaining}% (remainder)")
            else:
                allocation = self._get_decimal_input(f"TP{i+1} Allocation (%): ", 
                                                   min_value=Decimal('0.1'), max_value=Decimal('99.9'))
                tp.allocation_percentage = allocation
                total_allocation += allocation
    
    def _get_take_profit_levels(self, num_tps: int, entry: Decimal, direction: str) -> List[TakeProfitLevel]:
        """Get take profit levels with allocation presets or custom input."""
        # Display allocation preset options
        print("\nAllocation Presets:")
        print("1. Equal Split (25/25/25/25 for 4 levels)")
        print("2. Conservative (50/30/15/5)")
        print("3. Aggressive (20/30/30/20)")
        print("4. Custom (define each level)")
        
        preset_choice = self._get_integer_input("\nChoose allocation preset (1-4): ", min_value=1, max_value=4)
        
        # Collect TP prices using helper method
        take_profits = []
        print(f"\nEnter {num_tps} Take Profit levels:")
        for i in range(num_tps):
            tp_price = self._get_single_tp_price(i, entry, direction, take_profits)
            take_profits.append(TakeProfitLevel(price=tp_price, allocation_percentage=Decimal('0')))
        
        # Apply allocations using helper methods
        if preset_choice == 4:  # Custom
            self._get_custom_allocations(take_profits)
        else:  # Preset allocations
            self._apply_preset_allocations(take_profits, preset_choice, num_tps)
        
        return take_profits
        
    # ============================================================================================
    # RESULTS DISPLAY METHODS
    # ============================================================================================
    
    def display_comprehensive_analysis(self, setup: EnhancedTradeSetup, 
                                     position_result: PositionSizingResult, 
                                     risk_metrics: RiskMetrics, 
                                     scenarios: Dict[str, Dict]):
        """
        Display comprehensive analysis results in professional format.
        
        Args:
            setup: Validated trade setup
            position_result: Position sizing calculations
            risk_metrics: Risk assessment metrics
            scenarios: Scenario analysis results
        """
        print("\n" + "=" * 80)
        print("ZAWAD RISK ENGINE - ADVANCED POSITION SIZING ANALYSIS")
        print("=" * 80)
        
        self._display_trade_setup_analysis(setup)
        self._display_kelly_breakdown(position_result)
        self._display_position_sizing(position_result, setup)
        self._display_scenario_analysis(scenarios, position_result)
        self._display_risk_metrics(risk_metrics, setup)
        self._display_execution_plan(setup, position_result)
        self._display_risk_alerts(setup, position_result, risk_metrics)
    
    def _display_trade_setup_analysis(self, setup: EnhancedTradeSetup):
        """Display trade setup analysis section."""
        print("\nTRADE SETUP ANALYSIS:")
        print("-" * 25)
        
        # Calculate stop loss distance percentage
        sl_distance_pct = abs(setup.entry - setup.stop_loss) / setup.entry * Decimal('100')
        
        # Calculate weighted R-ratio
        weighted_r = self.calculate_weighted_rr_ratio(setup)
        
        print(f"    Direction: {setup.direction}")
        print(f"    Stop Loss Distance: {sl_distance_pct:.2f}%")
        print(f"    Weighted R-Ratio: {weighted_r:.4f}")
        
        print(f"    Individual R-Ratios: [{', '.join(f'{tp.individual_rr_ratio:.2f}' for tp in setup.take_profits)}]")
        
        # Risk assessment
        if weighted_r >= Decimal('2.0'):
            risk_assessment = "FAVORABLE"
        elif weighted_r >= Decimal('1.5'):
            risk_assessment = "ACCEPTABLE"
        else:
            risk_assessment = "MARGINAL"
        
        print(f"    Trade Risk Assessment: {risk_assessment}")
    
    def _display_kelly_breakdown(self, position_result: PositionSizingResult):
        """Display Kelly Criterion breakdown section."""
        print("\nKELLY CRITERION CALCULATIONS:")
        print("-" * 30)
        
        print(f"    Raw Kelly (Weighted): {position_result.kelly_raw:.4f} ({position_result.kelly_raw*100:.2f}%)")
        print(f"    Quarter-Kelly Applied: {position_result.kelly_quarter:.4f} ({position_result.kelly_quarter*100:.2f}%)")
        print(f"    Multi-TP Adjustment: {position_result.kelly_multi_tp_adjusted:.4f} ({position_result.kelly_multi_tp_adjusted*100:.2f}%)")
        print(f"    Confidence Scaled: {position_result.kelly_confidence_scaled:.4f} ({position_result.kelly_confidence_scaled*100:.2f}%)")
        
        if position_result.kelly_capped != position_result.kelly_confidence_scaled:
            print(f"    Kelly Capped (50% max): {position_result.kelly_capped:.4f} ({position_result.kelly_capped*100:.2f}%)")
        
        if position_result.final_risk_fraction != position_result.kelly_capped:
            print(f"    Hard Risk Capped (5% max): {position_result.final_risk_fraction:.4f} ({position_result.final_risk_fraction*100:.2f}%)")
        
        print(f"    Final Risk Allocation: {position_result.final_risk_fraction*100:.2f}%")
    
    def _display_position_sizing(self, position_result: PositionSizingResult, setup: EnhancedTradeSetup):
        """Display position sizing section."""
        print("\nPOSITION SIZING:")
        print("-" * 16)
        
        print(f"    Total Capital: ${setup.capital:,.2f}")
        print(f"    Capital at Risk: ${position_result.capital_at_risk:,.2f}")
        print(f"    Base Position: ${position_result.base_position_size:,.2f}")
        print(f"    Leveraged Position ({setup.leverage}x): ${position_result.leveraged_position_size:,.2f}")
        print(f"    Notional Exposure: ${position_result.notional_exposure:,.2f}")
    
    def _display_scenario_analysis(self, scenarios: Dict[str, Dict], position_result: PositionSizingResult):
        """Display scenario analysis table."""
        print("\nSCENARIO ANALYSIS:")
        print("-" * 18)
        
        # Sort scenarios by return percentage
        sorted_scenarios = sorted(scenarios.items(), key=lambda x: x[1]['return_percentage'])
        
        for scenario_name, scenario_data in sorted_scenarios:
            prob = scenario_data['probability'] * Decimal('100')
            return_pct = scenario_data['return_percentage']
            return_dollar = position_result.capital_at_risk * return_pct / Decimal('100')
            
            print(f"    {scenario_name:<15} ({prob:5.1f}%): {return_dollar:+8.2f} ({return_pct:+6.2f}%)")
    
    def _display_risk_metrics(self, risk_metrics: RiskMetrics, setup: EnhancedTradeSetup):
        """Display risk metrics dashboard."""
        print("\nRISK METRICS:")
        print("-" * 13)
        
        ev_pct = risk_metrics.expected_value / setup.capital * Decimal('100')
        
        print(f"    Expected Value: ${risk_metrics.expected_value:+.2f} ({ev_pct:+.2f}%)")
        print(f"    Probability of Profit: {risk_metrics.probability_of_profit*100:.1f}%")
        print(f"    Risk-Adjusted Return: {risk_metrics.risk_adjusted_score:.2f}")
        print(f"    Maximum Drawdown Impact: {risk_metrics.maximum_drawdown_impact:.2f}%")
        print(f"    Win Rate Breakeven: {risk_metrics.win_rate_breakeven*100:.1f}%")
        print(f"    R-Ratio Breakeven: {risk_metrics.r_ratio_breakeven:.2f}")
    
    def _display_execution_plan(self, setup: EnhancedTradeSetup, position_result: PositionSizingResult):
        """Display take profit execution plan with parallel allocation model."""
        print("\nTAKE PROFIT EXECUTION PLAN:")
        print("-" * 27)
        
        original_position = position_result.leveraged_position_size
        total_closed = Decimal('0')
        
        for i, tp in enumerate(setup.take_profits, 1):
            # Each TP closes a percentage of the ORIGINAL position (parallel allocation)
            close_amount = original_position * tp.allocation_percentage / Decimal('100')
            total_closed += close_amount
            remaining_position = original_position - total_closed
            
            print(f"    TP{i} (${tp.price}): Close {tp.allocation_percentage}% (${close_amount:,.2f}) -> Remaining: ${remaining_position:,.2f}")
    
    def _display_risk_alerts(self, setup: EnhancedTradeSetup, position_result: PositionSizingResult, risk_metrics: RiskMetrics):
        """Display risk management alerts."""
        print("\nRISK MANAGEMENT ALERTS:")
        print("-" * 23)
        
        alerts = []
        
        # Check for high leverage
        if setup.leverage > Decimal('20'):
            alerts.append(f"⚠️  HIGH LEVERAGE: {setup.leverage}x leverage increases volatility significantly")
        
        # Check for high capital risk
        if position_result.final_risk_fraction >= self.HARD_RISK_CAP:
            alerts.append(f"⚠️  MAXIMUM RISK: Using maximum allowed risk ({self.HARD_RISK_CAP*100}% of capital)")
        
        # Check for low probability of profit
        if risk_metrics.probability_of_profit < Decimal('0.60'):
            alerts.append(f"⚠️  LOW WIN PROBABILITY: Only {risk_metrics.probability_of_profit*100:.1f}% chance of profit")
        
        # Check for negative expected value
        if risk_metrics.expected_value < 0:
            alerts.append("🚨 NEGATIVE EXPECTED VALUE: This trade has negative expected return")
        
        if not alerts:
            alerts.append("✅ No significant risk alerts detected")
        
        for alert in alerts:
            print(f"    {alert}")
        
        print("\n" + "=" * 80)
    
    # ============================================================================================
    # MAIN EXECUTION METHODS
    # ============================================================================================
    
    def run_analysis(self):
        """
        Main execution method for the ZAWAD RISK ENGINE v3.9.
        
        Orchestrates the complete analysis flow:
        1. Input collection and validation
        2. Mathematical calculations
        3. Risk assessment
        4. Results display
        """
        try:
            # Phase 1: Input Collection
            setup = self.get_enhanced_trade_setup()
            
            # Phase 2: Validation
            print("\nSTEP 3: Validation")
            print("-" * 15)
            validation_result = self.validate_enhanced_inputs(setup)
            
            if not validation_result.is_valid:
                print(f"\n🚨 VALIDATION FAILED: {validation_result.error_message}")
                print("\nPlease correct the inputs and try again.")
                return
            
            print("✓ All validations passed")
            
            # Phase 3: Mathematical Calculations
            print("\nCalculating position sizing...")
            position_result = self.calculate_enhanced_kelly(setup)
            
            print("Performing risk assessment...")
            risk_metrics = self.calculate_risk_metrics(setup, position_result)
            
            print("Generating scenario analysis...")
            scenarios = self.calculate_scenario_matrix(setup)
            
            # Phase 4: Results Display
            self.display_comprehensive_analysis(setup, position_result, risk_metrics, scenarios)
            
        except KeyboardInterrupt:
            print("\n\nOperation cancelled by user.")
            sys.exit(0)
        
        except Exception as e:
            print(f"\n🚨 CRITICAL ERROR: {str(e)}")
            print("\nPlease check your inputs and try again.")
            sys.exit(1)


# ================================================================================================
# MAIN EXECUTION
# ================================================================================================

def main():
    """
    Entry point for the ZAWAD Risk Engine v3.9 CLI application.
    
    This function initializes the enhanced risk engine and starts the main execution loop.
    All mathematical calculations prioritize accuracy and capital preservation with comprehensive
    multi-level take profit analysis.
    """
    engine = ZawadRiskEngine()
    engine.run_analysis()


if __name__ == "__main__":
    main()
