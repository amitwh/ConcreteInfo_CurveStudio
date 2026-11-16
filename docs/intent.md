# ConcreteInfo CurveStudio
## Open-Source Scientific Curve Fitting & Engineering Analysis Platform

**Document:** Product Requirements and Technical Implementation Blueprint  
**Proposed version:** 1.0  
**Date:** November 2026  
**Primary language:** Python  
**Desktop framework:** PySide6 / Qt  
**Scientific engine:** NumPy, SciPy, LMfit, Statsmodels, SymPy  
**Web backend:** FastAPI  
**Proposed application licence:** GPL-3.0

---

# 1. Executive Summary

ConcreteInfo CurveStudio will be an open-source scientific data analysis, curve-fitting, regression, optimization and mathematical modelling platform.

The application will provide advanced capabilities for scientists, researchers, engineers, universities, laboratories and technical consultants.

The project will initially concentrate on curve fitting and regression but will evolve into a broader scientific analysis platform.

### Product objectives

1. Provide a comprehensive alternative to CurveExpert.
2. Support linear, polynomial and nonlinear regression.
3. Provide automatic model discovery and model comparison.
4. Support user-defined mathematical equations.
5. Offer comprehensive statistical and residual analysis.
6. Provide advanced scientific visualization.
7. Support interpolation, extrapolation, optimization and prediction.
8. Support multiple datasets and multivariable analysis.
9. Provide desktop and browser-based interfaces.
10. Enable domain-specific model extensions, particularly civil engineering and construction materials.

### Core product philosophy

- Scientific correctness over superficial functionality.
- Transparent mathematical algorithms.
- Reproducible calculations.
- Extensible architecture.
- No proprietary mathematical engine dependency.
- Offline desktop operation.
- Open formats for importing and exporting data.
- Consistent calculations across desktop and web.

# 2. Target Users

### Primary users

- Scientific researchers.
- Civil and structural engineers.
- Material scientists.
- Chemical engineers.
- Environmental engineers.
- Quality control laboratories.
- Academic institutions.
- Industrial R&D departments.
- Data analysts and technical consultants.

### Secondary users

- Students and educators.
- Statistical researchers.
- Manufacturing industries.
- Pharmaceutical and chemical laboratories.
- Energy and environmental researchers.

# 3. Product Scope

ConcreteInfo CurveStudio will have eight major functional areas.

| Module | Primary functionality |
|---|---|
| Data Studio | Data import, editing and preparation |
| Curve Fitting | Linear and nonlinear regression |
| Model Discovery | Automatic model selection |
| Statistical Analysis | Regression statistics and diagnostics |
| Visualization Studio | Interactive scientific plots |
| Mathematical Tools | Interpolation, smoothing and transformations |
| Prediction Studio | Estimation and prediction intervals |
| Engineering Models | Domain-specific engineering models |

The application should distinguish between an MVP, a professional release and a long-term advanced platform.

## 3.1 MVP

- CSV, XLSX and TXT imports.
- Data table and scatter plot.
- Linear regression.
- Polynomial regression.
- Common exponential, logarithmic and power models.
- Nonlinear least-squares fitting.
- R², adjusted R², RMSE, SSE and MAE.
- Equation display.
- Parameter estimation.
- Residual plots.
- PNG, SVG and CSV exports.
- Native desktop application.

## 3.2 Professional release

- 100+ model definitions.
- Automatic model discovery.
- Advanced nonlinear optimization.
- Parameter constraints.
- Weighted regression.
- Robust regression.
- Confidence and prediction intervals.
- Multiple datasets.
- Model comparison.
- Advanced residual diagnostics.
- Custom equations.
- Excel and PDF reporting.
- Browser-based application.

## 3.3 Advanced platform

- Multivariable regression.
- Global and simultaneous fitting.
- Multi-objective optimization.
- Bayesian parameter estimation.
- Bootstrap uncertainty analysis.
- 3D surface fitting.
- Differential-equation models.
- User-developed model plugins.
- Engineering-specific model libraries.
- Reproducible project workflows.

# 4. System Architecture

The most important architectural decision is to separate the scientific engine from the user interfaces.

Both desktop and web applications must call the same mathematical and statistical engine.

## 4.1 Logical architecture

```text
                    CURVELAB
                       |
           +-----------+-----------+
           |                       |
      DESKTOP CLIENT           WEB CLIENT
       PySide6 / Qt           Browser UI
           |                       |
           |                 FastAPI REST API
           |                       |
           +-----------+-----------+
                       |
                 APPLICATION CORE
                       |
       +---------------+----------------+
       |               |                |
   Data Engine     Model Engine     Statistics Engine
       |               |                |
       +---------------+----------------+
                       |
               COMPUTATION ENGINE
                       |
        +--------------+---------------+
        |              |               |
      NumPy          SciPy            LMfit
        |              |               |
        +--------------+---------------+
                       |
                RESULTS ENGINE
                       |
          Visualization / Reporting
```

## 4.2 Recommended technology stack

| Component | Technology |
|---|---|
| Core language | Python |
| Desktop interface | PySide6 |
| Desktop plotting | PyQtGraph / Matplotlib |
| Web backend | FastAPI |
| Web frontend | React + TypeScript |
| Web plotting | Plotly.js |
| Numerical computation | NumPy |
| Optimization | SciPy |
| Nonlinear fitting | LMfit |
| Statistical analysis | Statsmodels |
| Symbolic mathematics | SymPy |
| Data handling | Pandas |
| Excel support | OpenPyXL |
| Persistence | SQLite / SQLAlchemy |
| Web database | PostgreSQL |
| Serialization | Pydantic |
| Background processing | ProcessPoolExecutor initially |
| Testing | Pytest |
| Code quality | Ruff, MyPy |
| Documentation | MkDocs |
| Desktop packaging | PyInstaller |
| Web deployment | Docker |
| CI/CD | GitHub Actions |

Use compatible, pinned dependency versions and maintain a tested dependency matrix for Windows and Linux.

## 4.3 Proposed repository structure

```text
curvelab/
│
├── apps/
│   ├── desktop/
│   │   ├── main.py
│   │   ├── windows/
│   │   ├── dialogs/
│   │   ├── widgets/
│   │   ├── controllers/
│   │   └── resources/
│   │
│   ├── api/
│   │   ├── main.py
│   │   ├── routes/
│   │   ├── schemas/
│   │   └── middleware/
│   │
│   └── web/
│       ├── src/
│       ├── components/
│       ├── pages/
│       └── services/
│
├── packages/
│   ├── core/
│   │   ├── data/
│   │   ├── fitting/
│   │   ├── models/
│   │   ├── optimization/
│   │   ├── statistics/
│   │   ├── diagnostics/
│   │   ├── prediction/
│   │   ├── interpolation/
│   │   ├── transformations/
│   │   └── validation/
│   │
│   ├── visualization/
│   ├── reporting/
│   ├── engineering/
│   └── plugins/
│
├── model_library/
│   ├── linear/
│   ├── polynomial/
│   ├── exponential/
│   ├── logarithmic/
│   ├── power/
│   ├── sigmoid/
│   ├── statistical/
│   └── engineering/
│
├── tests/
│   ├── unit/
│   ├── integration/
│   ├── regression/
│   ├── numerical/
│   └── end_to_end/
│
├── examples/
├── documentation/
├── scripts/
├── deployment/
├── pyproject.toml
├── LICENSE
└── README.md
```

The core package must not depend on PySide6, React or FastAPI. This ensures the mathematical engine can be used independently.

# 5. Data Studio

Data handling is the foundation of the entire platform.

## 5.1 Import formats

Required:
- CSV
- XLSX
- XLS where supported
- TXT
- TSV
- JSON
- Manual data entry

Future:
- HDF5
- Parquet
- Scientific instrument exports
- Clipboard import

## 5.2 Data editor

Features:
- Spreadsheet-style editing.
- Multiple columns and datasets.
- Column naming.
- X/Y variable selection.
- Unit assignment.
- Missing-value identification.
- Duplicate identification.
- Outlier flagging.
- Sorting and filtering.
- Data transformations.
- Undo and redo.

## 5.3 Data validation

The system should detect:
- Non-numeric values.
- Missing values.
- Infinite values.
- Duplicate observations.
- Insufficient sample size.
- Constant X variables.
- Zero and negative values in logarithmic transformations.
- Invalid weight values.

Never silently delete observations. All cleaning operations must be recorded and reversible.

# 6. Curve Fitting Engine

This is the central component of ConcreteInfo CurveStudio.

## 6.1 Model categories

### A. Linear models

- Simple linear regression.
- Multiple linear regression.
- Weighted linear regression.
- Robust linear regression.
- Through-origin regression.

### B. Polynomial models

- Quadratic.
- Cubic.
- Quartic.
- Fifth through tenth order.
- User-defined polynomial order.

High-order polynomial models should include warnings about instability and extrapolation.

### C. Exponential models

Examples:

y = a exp(bx)

y = a exp(-bx) + c

y = a + b exp(cx)

Double exponential models.

### D. Logarithmic models

y = a + b ln(x)

y = a + b log10(x)

Modified logarithmic relationships.

### E. Power models

y = ax^b

Generalized power relationships.

### F. Saturation models

- Michaelis-Menten.
- Monod.
- Asymptotic.
- Langmuir.
- Exponential saturation.

### G. Sigmoidal models

- Logistic.
- Four-parameter logistic.
- Five-parameter logistic.
- Gompertz.
- Weibull.
- Richards.

### H. Statistical distribution models

- Gaussian.
- Lorentzian.
- Voigt.
- Cauchy.
- Lognormal.
- Gamma.
- Weibull distributions.

### I. Interpolation and smoothing

- Linear interpolation.
- Cubic spline.
- PCHIP.
- Akima interpolation.
- Moving average.
- LOWESS.
- Savitzky-Golay.
- Gaussian smoothing.

Interpolation and smoothing must be identified separately from regression fitting in the user interface.

## 6.2 Model definition framework

Every model should follow a common interface.

A model definition must contain:

- Unique model identifier.
- Display name.
- Mathematical equation.
- Independent variables.
- Parameter names.
- Parameter descriptions.
- Initial parameter estimates.
- Parameter bounds.
- Model evaluator.
- Optional analytical derivatives.
- Parameter constraints.
- Model category.
- Documentation and references.
- Domain-specific validation rules.

Illustrative interface:

```python
class BaseModel:
    model_id: str
    name: str
    parameters: list

    def evaluate(self, x, params):
        raise NotImplementedError

    def initial_guess(self, x, y):
        raise NotImplementedError

    def parameter_bounds(self):
        raise NotImplementedError
```

All models should be independently testable.

# 7. Numerical Algorithms

The application should use established numerical algorithms instead of developing untested optimizers.

## 7.1 Least-squares methods

Implement:

1. Ordinary least squares.
2. Weighted least squares.
3. Nonlinear least squares.
4. Robust least squares.
5. Orthogonal distance regression.

Recommended implementation:

- NumPy for numerical arrays.
- SciPy optimize for optimization.
- LMfit for parameter management and fitting workflows.
- SciPy ODR for errors in both X and Y where appropriate.

## 7.2 Optimization algorithms

Support:

- Levenberg-Marquardt.
- Trust Region Reflective.
- Dogbox.
- Nelder-Mead.
- Powell.
- BFGS.
- L-BFGS-B.
- Differential Evolution.
- Dual Annealing.

Not every optimizer is appropriate for every problem. The application must validate method compatibility with bounds, derivatives and constraints.

## 7.3 Initial parameter estimation

Poor starting estimates are one of the biggest causes of failed nonlinear fitting.

Provide:
- Analytical initial estimates where possible.
- Linearized estimates where mathematically valid.
- Data-derived heuristics.
- User-supplied initial values.
- Multi-start optimization.
- Randomized starting points with reproducible seeds.

Store the optimizer, initial parameters, final parameters, convergence status and termination reason.

## 7.4 Model comparison

Model ranking must not depend exclusively on R².

Calculate:

- SSE.
- MSE.
- RMSE.
- MAE.
- R².
- Adjusted R² where applicable.
- AIC.
- AICc.
- BIC.
- Residual standard error.
- Cross-validation error.

Information criteria must use a clearly documented likelihood and parameter-count convention.

Do not compare scores calculated using incompatible datasets or different response transformations without warning.

# 8. Statistical Analysis and Diagnostics

This module distinguishes a professional scientific application from a basic curve-fitting utility.

## 8.1 Regression statistics

For every fitting operation, provide:

- Parameter estimates.
- Standard errors.
- Parameter covariance matrix.
- Parameter correlation matrix.
- Confidence intervals.
- Residual standard deviation.
- Goodness-of-fit statistics.
- Degrees of freedom.
- Convergence diagnostics.

## 8.2 Residual analysis

Required visualizations:

1. Residual versus fitted values.
2. Residual versus X.
3. Normal Q-Q plot.
4. Residual histogram.
5. Scale-location plot.
6. Autocorrelation plot where appropriate.
7. Standardized residuals.

## 8.3 Statistical tests

Potential tests:

- Shapiro-Wilk normality test.
- Anderson-Darling test.
- Breusch-Pagan heteroscedasticity test.
- Ljung-Box autocorrelation test.
- Durbin-Watson statistic.
- Cook's distance.
- Leverage analysis.

The application must communicate the assumptions and limitations of each test. A statistical test should never automatically certify a model as scientifically valid.

## 8.4 Uncertainty analysis

Professional release:

- Delta-method confidence intervals.
- Bootstrap parameter estimation.
- Monte Carlo uncertainty propagation.
- Profile likelihood intervals where supported.
- Prediction intervals.
- Confidence bands.

Distinguish confidence intervals for the fitted mean response from prediction intervals for individual future observations.

# 9. Automatic Model Discovery

This should become one of ConcreteInfo CurveStudio's flagship capabilities.

## Workflow

1. User imports data.
2. User selects independent and dependent variables.
3. System checks data validity.
4. System identifies applicable model families.
5. System estimates initial parameters.
6. System fits candidate models.
7. System validates convergence.
8. System calculates statistical metrics.
9. System performs residual diagnostics.
10. System ranks suitable models.
11. User inspects and selects the final model.

## Model ranking

Use a configurable ranking system based on:

- AICc.
- BIC.
- Cross-validation RMSE.
- Adjusted R².
- Residual quality.
- Model complexity.

Avoid automatically selecting high-order models simply because they achieve a higher training R².

Provide a model comparison table with:
- Model name.
- Equation.
- Parameters.
- R².
- RMSE.
- AICc.
- BIC.
- Validation score.
- Convergence status.

Allow users to inspect the top-ranked models rather than receiving only one answer.

# 10. Custom Equation Builder

Users should be able to define their own equations.

Example:

```text
y = a * exp(-b*x) + c
```

Supported functions:

- sin, cos, tan.
- exp, log, log10.
- sqrt.
- abs.
- sinh, cosh.
- pow.
- min, max where mathematically supported.

Features:

- Named parameters.
- Initial values.
- Bounds.
- Parameter constraints.
- Multiple independent variables.
- Equation validation.
- Symbolic differentiation where possible.
- Equation export to LaTeX.
- Safe numerical evaluation.

IMPORTANT: Never execute arbitrary user-provided Python with eval or exec. Parse expressions into a restricted, validated mathematical representation and expose only approved functions.

# 11. Multivariable Analysis

This should be a major differentiator over simple curve-fitting applications.

Support:

- Multiple linear regression.
- Nonlinear multivariable regression.
- Interaction terms.
- Polynomial features.
- Response surface methodology.
- Two-dimensional and three-dimensional surface fitting.
- Regularized regression.
- Ridge regression.
- Lasso.
- Elastic Net.
- Principal component regression.

Potential future functionality:

- Partial least squares.
- Generalized additive models.
- Generalized linear models.
- Mixed-effects models.
- Gaussian process regression.

Provide clear separation between explanatory regression, predictive modelling and causal inference.

# 12. Optimization Studio

Add a dedicated optimization module.

Capabilities:

- Single-variable optimization.
- Multivariable optimization.
- Parameter estimation.
- Constrained optimization.
- Multi-objective optimization.
- Sensitivity analysis.
- Parameter sweep.
- Response surface optimization.

Users should be able to define:
- Objective function.
- Independent variables.
- Parameter bounds.
- Constraints.
- Optimization direction.
- Initial conditions.

Engineering applications can later use this module for mix optimization, material proportioning and cost-performance optimization.

# 13. Scientific Visualization Studio

## Plot types

- Scatter plots.
- Line plots.
- Regression curves.
- Multiple fitted curves.
- Residual plots.
- Histograms.
- Box plots.
- Probability plots.
- Contour plots.
- Heatmaps.
- 3D scatter plots.
- 3D surface plots.
- Error-bar plots.
- Confidence bands.
- Prediction bands.

## Plot customization

- Axis labels.
- Units.
- Logarithmic axes.
- Linear axes.
- Grid controls.
- Legends.
- Annotation.
- Equation display.
- Parameter display.
- Multiple Y axes where justified.
- Custom line styles.
- Export resolution.

## Export

- PNG.
- JPEG.
- SVG.
- PDF.
- CSV data.

Desktop plotting should use PyQtGraph for responsive interaction and Matplotlib for publication-quality output. The web application should use Plotly.js.

# 14. Prediction Studio

Users should be able to use a fitted model as a prediction calculator.

Features:

- Single-point prediction.
- Batch prediction.
- Prediction tables.
- Interpolation.
- Extrapolation.
- Confidence intervals.
- Prediction intervals.
- Sensitivity analysis.

The application must clearly flag extrapolation outside the observed data range.

For nonlinear models, prediction uncertainty should account for parameter uncertainty and, where relevant, observation error.

# 15. Engineering Models Module

This module is particularly important for the long-term product strategy.

Create separate model libraries for engineering disciplines.

## Civil engineering and construction materials

Initial model categories:

### Concrete strength

- Compressive strength versus age.
- Exponential strength development.
- Hyperbolic strength development.
- Logistic strength development.
- Strength versus water/binder ratio.
- Strength versus cementitious content.
- Strength versus supplementary cementitious material replacement.

### Concrete properties

- Slump versus admixture dosage.
- UPV versus compressive strength.
- Elastic modulus versus compressive strength.
- Shrinkage development.
- Creep development.
- Maturity versus strength.
- Temperature versus strength development.

### Cement and supplementary cementitious materials

- Cement strength development.
- Fly ash replacement relationships.
- GGBS replacement relationships.
- Pozzolanic reactivity relationships.
- Hydration heat development.

### Durability

- Chloride diffusion models.
- Carbonation depth development.
- Water absorption relationships.
- Sorptivity.
- Permeability relationships.

### Particle packing

- Particle size distribution.
- Fuller and modified Fuller curves.
- Andreasen-Andersen distribution.
- Particle packing optimization.

These models should be presented as empirical or theoretical models with clearly documented assumptions. They must not be presented as universal IS-code acceptance equations unless a relevant standard explicitly establishes that relationship.

# 16. User Interface Specification

The UI should resemble a professional scientific desktop application, not a generic CRUD application.

## 16.1 Main desktop workspace

Proposed layout:

```text
+----------------------------------------------------------+
| ConcreteInfo CurveStudio     File  Edit  Data  Analysis  View  Help      |
+----------------------------------------------------------+
| New | Open | Save | Import | Fit | Predict | Export      |
+----------------+-------------------------+---------------+
| PROJECT        |                         | ANALYSIS      |
|                |                         |               |
| Dataset 1      |       PLOT AREA          | Model         |
| Dataset 2      |                         | Parameters    |
| Dataset 3      |     Scatter + Fit        | Statistics    |
|                |                         | Diagnostics   |
|                |                         |               |
+----------------+-------------------------+---------------+
| DATA TABLE                                                |
| X          Y          Fitted      Residual                |
|----------------------------------------------------------|
| STATUS: Model converged | Observations: 45               |
+----------------------------------------------------------+
```

## 16.2 Navigation

Main modules:

1. Home.
2. Data Studio.
3. Curve Fitting.
4. Model Discovery.
5. Statistical Analysis.
6. Visualization Studio.
7. Prediction Studio.
8. Optimization Studio.
9. Engineering Models.
10. Settings.

## 16.3 Model fitting panel

The fitting panel should include:

- Model selection.
- Equation preview.
- Parameter editor.
- Initial guesses.
- Parameter bounds.
- Fitting method.
- Weighting method.
- Convergence settings.
- Run fit button.

## 16.4 Results panel

Use tabbed results:

- Overview.
- Equation.
- Parameters.
- Statistics.
- Residuals.
- Diagnostics.
- Prediction.
- Report.

## 16.5 Project management

A ConcreteInfo CurveStudio project must preserve:

- Original imported datasets.
- Cleaned datasets.
- Data transformations.
- Selected variables.
- Model definitions.
- Parameter estimates.
- Fitting configuration.
- Statistical results.
- Plot configurations.
- Report settings.
- Application and dependency versions.

Use a versioned, portable project format such as `.clab`, implemented as a ZIP container with a JSON manifest and data files. Avoid unsafe Python object serialization.

# 17. Web Application Specification

The web interface should share the same core computation engine.

## Web architecture

```text
Browser
   |
React + TypeScript
   |
FastAPI
   |
Application Services
   |
ConcreteInfo CurveStudio Core
   |
SciPy / LMfit / NumPy
   |
Results
```

## Web functionality

The web application should support:

- User registration and authentication.
- Dataset upload.
- Data editing.
- Interactive plotting.
- Curve fitting.
- Model discovery.
- Statistical diagnostics.
- Project saving.
- Report generation.
- Project download.

## Backend considerations

Use:
- FastAPI.
- Pydantic validation.
- SQLAlchemy.
- PostgreSQL.
- Redis only when needed for caching or job coordination.
- Background worker processes for expensive computations.

Do not run computationally intensive optimization directly in the web request process.

Initially, use a controlled process pool and job table. Introduce a dedicated distributed task queue only when workload justifies it.

The API should have versioned endpoints:

```text
/api/v1/projects
/api/v1/datasets
/api/v1/models
/api/v1/fitting
/api/v1/discovery
/api/v1/statistics
/api/v1/predictions
/api/v1/optimization
/api/v1/reports
```

# 18. Reporting and Reproducibility

Every analysis should be reproducible.

A report should include:

1. Project name.
2. Dataset description.
3. Number of observations.
4. Model equation.
5. Fitting method.
6. Parameter estimates.
7. Parameter confidence intervals.
8. Regression statistics.
9. Residual plots.
10. Model comparison.
11. Prediction results.
12. Warnings and assumptions.
13. Software version.
14. Calculation date.
15. Dataset fingerprint.

Provide:
- PDF reports.
- Excel workbooks.
- CSV results.
- SVG/PNG graphs.
- Machine-readable JSON results.

# 19. Security and Reliability

Security requirements:

- Safe equation parsing.
- No arbitrary Python execution.
- Validated file uploads.
- File size limits.
- Archive extraction safeguards.
- API authentication.
- Per-user resource limits.
- Job timeouts.
- Rate limiting.
- Secure project storage.
- Database backups.

Reliability requirements:

- Deterministic calculation options.
- Recorded random seeds.
- Explicit convergence status.
- Numerical error handling.
- Graceful failure for invalid datasets.
- No silent replacement of failed fits with plausible-looking results.

# 20. Testing Strategy

Testing is a core product requirement, not a final-stage activity.

## Unit testing

Test:
- Every model equation.
- Parameter initialization.
- Parameter bounds.
- Fitting routines.
- Statistical calculations.
- Residual calculations.
- Prediction intervals.
- Data import.
- Data transformations.

## Numerical validation

Generate synthetic datasets using known mathematical equations and known parameters.

Example:

```python
import numpy as np

x = np.linspace(0, 10, 100)
true_y = 5 * np.exp(-0.4 * x) + 2

noise = np.random.default_rng(42).normal(
    0, 0.1, len(x)
)

y = true_y + noise
```

Fit the exponential model and verify that the recovered parameters are within predefined tolerances.

## Integration testing

Verify:
- CSV import to fitting.
- Excel import to fitting.
- Desktop project save and reload.
- API fitting requests.
- Web project persistence.
- PDF report generation.
- Cross-platform consistency.

## Performance testing

Benchmark:
- 100 observations.
- 1,000 observations.
- 10,000 observations.
- 100,000 observations.

Test multiple model families, multiple starting points and multiple simultaneous fitting jobs.

Performance targets should be benchmarked on documented hardware rather than treated as universal guarantees.

## Scientific verification

Use independent reference calculations and synthetic datasets with known solutions. For selected methods, compare results with trusted scientific packages.

Maintain regression tests so numerical changes do not silently alter previously validated results.

# 21. Deployment Strategy

## Desktop

Target operating systems:

- Windows 10/11.
- Ubuntu LTS.
- Other mainstream Linux distributions where dependencies permit.

Packaging:

- Windows installer.
- Portable Windows build.
- Linux AppImage.
- Linux package formats as demand develops.

The desktop version must work offline after installation.

## Web

Docker-based deployment:

```text
Reverse Proxy
     |
Frontend Container
     |
FastAPI Container
     |
Computation Workers
     |
PostgreSQL
```

Recommended deployment options:

- Docker Compose for simple installations.
- Coolify for self-hosted deployment.
- PostgreSQL for persistent data.
- Persistent storage for datasets and projects.
- Automated backups.
- HTTPS through the reverse proxy.

Keep the desktop build independent of the web deployment.

# 22. Development Roadmap

A proposed 20-week development programme.

| Phase | Duration | Deliverable |
|---|---:|---|
| 0. Foundation | 1 week | Repository, architecture, CI |
| 1. Data Studio | 2 weeks | Import, editing, validation |
| 2. Core fitting | 3 weeks | Linear, polynomial, nonlinear |
| 3. Model library | 3 weeks | 50+ initial models |
| 4. Statistics | 2 weeks | Diagnostics and uncertainty |
| 5. Desktop UI | 3 weeks | Integrated desktop application |
| 6. Visualization/reporting | 2 weeks | Advanced graphs and exports |
| 7. Web platform | 3 weeks | FastAPI and browser UI |
| 8. Validation/release | 1 week | Testing and packaging |

This is a proposed development schedule for a focused team with parallel work where possible. A single developer should expect a longer timeline.

## Release milestones

### Alpha
- Data import.
- Basic regression.
- Initial desktop interface.
- Core numerical tests.

### Beta
- Nonlinear fitting.
- Model discovery.
- Statistical diagnostics.
- Advanced visualization.
- Project persistence.

### Release Candidate
- Web application.
- Reporting.
- Packaging.
- Security testing.
- Documentation.

### Version 1.0
- Stable desktop application.
- Stable web application.
- Published model library.
- Reproducible projects.
- Installation packages.
- Documentation and tutorials.

# 23. Team and Workstream Organization

Recommended workstreams:

| Workstream | Responsibilities |
|---|---|
| Scientific engine | Algorithms, models, numerical validation |
| Desktop application | PySide6 UI, project management |
| Web application | React, FastAPI, authentication |
| Visualization | Interactive and publication-quality plots |
| QA and documentation | Tests, examples, tutorials, scientific verification |

For a small team, these can be parallel responsibilities rather than separate full-time positions.

# 24. Open-Source Governance

Recommended licence:

- Application: GPL-3.0.
- Independent reusable mathematical libraries: MIT or BSD-3-Clause where appropriate.
- Documentation: CC BY 4.0.

Repository requirements:

- Contributor guidelines.
- Code of conduct.
- Issue templates.
- Pull request templates.
- Versioning policy.
- Release notes.
- Scientific model contribution guidelines.
- Automated CI tests.

Every contributed mathematical model must include its equation, parameter definitions, references, tests and applicable assumptions.

# 25. Key Risks and Mitigation

| Risk | Mitigation |
|---|---|
| Poor nonlinear convergence | Multi-start, bounds, robust initialization |
| Misleading model rankings | AICc, validation, complexity penalties |
| Incorrect statistical interpretation | Explicit assumptions and warnings |
| Inconsistent desktop/web results | Shared core engine |
| Performance limitations | Background processes and profiling |
| Unsafe custom equations | Restricted expression parser |
| Scope expansion | Strict MVP and release gates |
| Model library inaccuracies | Peer review and reference tests |

# 26. Success Criteria

Version 1.0 should meet the following acceptance criteria:

1. Runs on Windows and Linux.
2. Provides offline desktop functionality.
3. Supports the core scientific model families.
4. Supports user-defined equations safely.
5. Provides nonlinear parameter estimation.
6. Produces reliable statistical diagnostics.
7. Supports automatic model comparison.
8. Exports scientific graphs and reports.
9. Preserves reproducible project files.
10. Exposes equivalent core functionality through a web application.
11. Passes automated unit, integration and numerical validation tests.
12. Has complete installation and user documentation.

# 27. Long-Term Product Vision

ConcreteInfo CurveStudio should eventually become a modular scientific analysis platform.

Potential extensions:

- Time-series analysis.
- Signal processing.
- Differential equations.
- Parameter estimation.
- Experimental design.
- Design of experiments.
- Bayesian modelling.
- Machine learning regression.
- Scientific image analysis.
- Materials science models.
- Structural engineering models.
- Chemical process models.

The scientific core should remain independently reusable.

The engineering libraries should be optional modules rather than hardcoded into the central regression engine.

---

# Final Product Strategy

Build the scientific engine first, desktop second and web third.

The desktop application establishes the complete scientific workflow and allows calculations to work offline. The web application then reuses the validated core through FastAPI.

The most important competitive differentiators should be:

1. Powerful nonlinear fitting.
2. Automatic model discovery.
3. Transparent statistical diagnostics.
4. Safe custom equation creation.
5. Multivariable scientific modelling.
6. Reproducible analysis projects.
7. Domain-specific engineering libraries.
8. A genuinely open and extensible model ecosystem.

**The intended result is not simply an open-source CurveExpert replacement. It is an extensible scientific and engineering analysis platform that can grow independently of any one application or discipline.**
