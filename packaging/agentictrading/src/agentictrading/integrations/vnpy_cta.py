"""Public, lazily extensible API for the vn.py CTA integration."""

from ._vnpy_cta_core import (
    ARTIFACT_SCHEMA_VERSION,
    AUDIT_STATUSES,
    ArtifactValidationError,
    CapturedCtaOrder,
    CtaOrderMapping,
    VnpyCtaAuditArtifact,
    VnpyCtaAuditRecord,
    build_audit_artifact,
    build_safe_manifest,
    load_audit_artifact,
    map_captured_order,
    sanitize_error_message,
    save_audit_artifact,
)
from ._vnpy_cta_runtime import (
    AtlCtaEngine,
    VnpyBindings,
    VnpyCtaCompatibilityError,
    VnpyCtaDependencyError,
    VnpyCtaRuntime,
    load_vnpy_bindings,
)
from ._vnpy_cta_adapter import VnpyCtaAdapter, VnpyCtaDataError
from ._vnpy_cta_runner import VnpyCtaATLRunner, VnpyCtaRunSummary

__all__ = [
    "ARTIFACT_SCHEMA_VERSION",
    "AUDIT_STATUSES",
    "ArtifactValidationError",
    "CapturedCtaOrder",
    "CtaOrderMapping",
    "VnpyCtaAuditArtifact",
    "VnpyCtaAuditRecord",
    "map_captured_order",
    "build_safe_manifest",
    "sanitize_error_message",
    "build_audit_artifact",
    "save_audit_artifact",
    "load_audit_artifact",
    "VnpyCtaDependencyError",
    "VnpyCtaCompatibilityError",
    "VnpyBindings",
    "load_vnpy_bindings",
    "AtlCtaEngine",
    "VnpyCtaRuntime",
    "VnpyCtaDataError",
    "VnpyCtaAdapter",
    "VnpyCtaRunSummary",
    "VnpyCtaATLRunner",
]
