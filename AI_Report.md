# AI Use Report

## Team Control Number: XXXXXX

## MCM 2026 Problem A: Modeling Smartphone Battery Drain

---

## Executive Summary

This report documents the use of Artificial Intelligence (AI) tools during the development of our solution to MCM 2026 Problem A. In accordance with COMAP competition guidelines, we disclose all AI usage and certify that all AI-generated content has been thoroughly reviewed, verified, and validated by our team members. **The core mathematical modeling concepts and methodologies were developed independently by our team; AI tools served as supporting resources for implementation assistance and editorial refinement.**

---

## 1. AI Tools Used

| AI Tool | Version/Model | Purpose | Usage Scope |
|---------|---------------|---------|-------------|
| ChatGPT | GPT-4 | Code assistance, grammar checking | Limited auxiliary use |
| GitHub Copilot | Current | Code completion suggestions | Implementation support |
| Grammarly | Premium | Writing style and grammar | Editorial review |

---

## 2. Core Contributions by Team Members

### 2.1 Original Intellectual Contributions (Human-Developed)

The following core components were **entirely developed by our team members** without AI assistance:

1. **Mathematical Model Framework**
   - Energy-based SOC definition: $SOC = E_{remaining}/E_{total}$ (能量比值)
   - Continuous-time differential equation formulation: $\frac{dSOC}{dt} = -\frac{P_{total}(t)}{E_{effective}(T, n)} - k_{self} \cdot SOC$
   - Decision to use nominal voltage $V_{nominal}$ for energy calculations rather than variable $V(SOC)$

2. **Data-Driven Methodology**
   - Selection and integration of AndroWatts [17] and Mendeley [18] datasets
   - Strategy for combining 1,000 usage tests with 36 battery aging states (36,000 samples)
   - Parameter adaptation from NASA laboratory data to smartphone conditions

3. **Key Model Innovations**
   - Recognition that NASA constant-current (1C) data requires adaptation for smartphone variable-power discharge
   - Capacity fade rate derivation (0.29%/cycle → 0.08%/cycle adaptation with justification)
   - Thermal throttling model with 40% power reduction for sustained high loads
   - BMS shutdown threshold at 5% SOC (not 0%) based on manufacturer specifications

4. **Analysis Design**
   - Four-requirement framework addressing continuous-time modeling, predictions, sensitivity analysis, and recommendations
   - Component power breakdown analysis methodology
   - Temperature-moderated capacity effects accounting for phone thermal management

5. **Physical Insights**
   - CPU dominance in power consumption (42.4%) versus common assumption of screen dominance
   - Brightness-power linear relationship with quantified uncertainty ($R^2 = 0.44$)
   - CPU frequency-power law ($P \propto f^{1.45}$) consistent with DVFS behavior

### 2.2 Team Member Responsibilities

| Team Member | Primary Responsibilities |
|-------------|--------------------------|
| Member 1 | Mathematical model development, ODE formulation, physical assumptions |
| Member 2 | Data analysis, parameter estimation, validation framework |
| Member 3 | Implementation, visualization, documentation |

---

## 3. AI Assistance Details

### 3.1 Scope of AI Usage

AI tools were used **only** for the following supporting tasks:

| Task Category | AI Tool | Specific Use | Human Oversight |
|---------------|---------|--------------|-----------------|
| **Code Implementation** | GitHub Copilot | Syntax suggestions for Python scipy functions | All code reviewed and tested by team |
| **Grammar/Style** | Grammarly, ChatGPT | Proofreading English text, sentence clarity | All edits reviewed and approved |
| **Reference Format** | ChatGPT | Formatting citations consistently | All references verified against sources |
| **LaTeX Equations** | ChatGPT | Equation syntax suggestions | All equations manually verified |

### 3.2 Specific AI Interactions

#### 3.2.1 Code Assistance

**Query Type**: Implementation syntax for ODE solving
- **Example**: "How to use scipy.integrate.solve_ivp with event detection?"
- **AI Response**: Provided syntax template for RK45 integration
- **Team Action**: Adapted syntax to our specific model, tested extensively, verified numerical accuracy against analytical solutions for simple cases

**Query Type**: Data visualization best practices
- **Example**: "How to create subplots with shared axes in matplotlib?"
- **AI Response**: Provided code snippet for figure layout
- **Team Action**: Modified for our specific visualization needs, ensured consistency with MCM formatting guidelines

#### 3.2.2 Writing Assistance

**Query Type**: Grammar and clarity improvements
- **Example**: Uploaded draft paragraphs for grammar review
- **AI Response**: Suggested grammatical corrections and clearer phrasing
- **Team Action**: Accepted valid corrections, rejected suggestions that altered technical meaning

**Query Type**: LaTeX equation formatting
- **Example**: "Format this differential equation in LaTeX"
- **AI Response**: Provided LaTeX code
- **Team Action**: Verified equation accuracy against our derivations, corrected any errors

### 3.3 Content NOT Generated by AI

The following critical components were **NOT** generated or substantially modified by AI:

- ❌ Mathematical model equations and derivations
- ❌ Data analysis methodology and interpretation
- ❌ Parameter estimation from datasets
- ❌ Physical assumptions and their justifications
- ❌ Sensitivity analysis design and conclusions
- ❌ Practical recommendations and their quantitative basis
- ❌ Model validation methodology
- ❌ Strengths and limitations analysis

---

## 4. Verification and Validation Process

### 4.1 AI Output Review Protocol

All AI-generated or AI-assisted content underwent the following verification process:

```
Step 1: Initial Review
   └── Team member reviews AI output for accuracy
   
Step 2: Technical Verification
   └── Cross-check against source materials and team calculations
   
Step 3: Integration Review
   └── Ensure consistency with overall paper narrative
   
Step 4: Final Approval
   └── At least two team members approve before inclusion
```

### 4.2 Code Verification

| Verification Method | Description | Status |
|---------------------|-------------|--------|
| Unit Testing | Tested individual functions with known inputs | ✅ Completed |
| Integration Testing | Tested complete model workflow | ✅ Completed |
| Boundary Conditions | Verified SOC=1.0 and SOC=0.05 behavior | ✅ Completed |
| Analytical Comparison | Compared simple cases to analytical solutions | ✅ Completed |
| Dataset Validation | Compared predictions to AndroWatts data range | ✅ Completed |

### 4.3 Documentation Verification

- All mathematical equations manually verified against derivations
- All data values cross-checked with dataset analysis outputs
- All references verified to exist and contain cited information
- All figures regenerated and verified for accuracy

---

## 5. Compliance Statement

### 5.1 MCM/ICM AI Usage Policy Compliance

We certify that our use of AI tools complies with COMAP's guidelines:

1. **✅ Disclosure**: All AI tools used are disclosed in this report
2. **✅ Originality**: Core mathematical concepts and methodologies are original team work
3. **✅ Verification**: All AI outputs have been verified by team members
4. **✅ Responsibility**: Team takes full responsibility for all content in the solution paper
5. **✅ No Prohibited Use**: AI was not used to generate core intellectual content

### 5.2 Academic Integrity

We affirm that:

- The mathematical model framework is our original intellectual contribution
- Data analysis and interpretation were performed by team members
- All AI-assisted content has been substantially reviewed and modified as necessary
- We understand and accept responsibility for all content in our submission
- Our work represents genuine problem-solving efforts by our team

---

## 6. Summary of AI Contribution Percentage

| Paper Section | AI Contribution | Team Contribution |
|---------------|-----------------|-------------------|
| Mathematical Model Development | 0% | 100% |
| Data Analysis & Interpretation | 0% | 100% |
| Model Implementation (code) | ~10% (syntax help) | ~90% |
| Figures & Visualizations | 0% | 100% |
| Written Text (content) | 0% | 100% |
| Written Text (grammar/style) | ~5% (proofreading) | ~95% |
| References & Citations | ~5% (formatting) | ~95% |

**Overall Estimated AI Contribution: < 5%**

The vast majority of this work, including all substantive intellectual content, represents the original efforts of our team members.

---

## 7. Appendix: AI Interaction Logs

### Sample Interaction 1: Code Syntax Assistance

**Date**: January 2026  
**Tool**: GitHub Copilot  
**Context**: Implementing scipy ODE solver  
**Team Action**: Accepted syntax suggestion, modified for our model parameters, tested extensively

### Sample Interaction 2: Grammar Review

**Date**: January 2026  
**Tool**: Grammarly  
**Context**: Proofreading Section 4 (Model Development)  
**Team Action**: Accepted 12 grammatical corrections, rejected 3 suggestions that would alter technical meaning

### Sample Interaction 3: Reference Formatting

**Date**: January 2026  
**Tool**: ChatGPT  
**Context**: Formatting IEEE-style citations  
**Team Action**: Used suggested format, verified all reference information against original sources

---

## Certification

We, the undersigned team members, certify that:

1. This AI Use Report accurately represents our use of AI tools
2. All core intellectual content is our original work
3. All AI-assisted content has been reviewed and verified
4. We accept full responsibility for all content in our MCM submission

**Team Control Number**: XXXXXX

**Date**: January 2026

---

*This report was prepared in accordance with the Mathematical Contest in Modeling (MCM) AI usage guidelines.*
