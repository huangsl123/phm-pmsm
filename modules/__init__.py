# -*- coding: utf-8 -*-
"""
Modules for domain adaptation
"""

from .domain_adaptation import (
    MultipleKernelMaximumMeanDiscrepancy,
    DomainDiscriminator,
    GradientReverseLayer,
    DomainAdaptationModule,
    get_default_domain_adaptation_module,
    CoralLoss
)

__all__ = [
    'MultipleKernelMaximumMeanDiscrepancy',
    'DomainDiscriminator',
    'GradientReverseLayer',
    'DomainAdaptationModule',
    'get_default_domain_adaptation_module',
    'CoralLoss'
]
