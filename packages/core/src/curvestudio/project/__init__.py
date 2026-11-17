"""The ``.cstudio`` project format: datasets, analyses and plot settings in one ZIP."""

from curvestudio.project.archive import load_project, save_project
from curvestudio.project.errors import ProjectError, ProjectErrorCode
from curvestudio.project.model import Analysis, Manifest, ManifestEntry, Project, ProjectDataset

__all__ = [
    "Analysis",
    "Manifest",
    "ManifestEntry",
    "Project",
    "ProjectDataset",
    "ProjectError",
    "ProjectErrorCode",
    "load_project",
    "save_project",
]
