# AI Use Report

## Team Control Number: XXXXXX

## MCM 2026 Problem A: Modeling Smartphone Battery Drain

---

## Executive Summary

This report documents the use of Artificial Intelligence (AI) tools during the development of our solution to MCM 2026 Problem A. In accordance with COMAP competition guidelines, we disclose all AI usage and certify that all AI-generated content has been thoroughly reviewed, verified, and validated by our team members. The core modeling concepts and methodology were developed by our team, while AI tools provided valuable assistance in implementation, writing refinement, and literature research.

---

## 1. AI Tools Used

| AI Tool | Version/Model | Purpose | Usage Scope |
|---------|---------------|---------|-------------|
| ChatGPT | GPT-4 | Code development, literature research, writing assistance | Substantial use throughout project |
| Claude | Claude 3 | Model validation, equation verification | Analysis support |
| GitHub Copilot | Current | Code completion and generation | Implementation |
| Grammarly | Premium | Grammar and style checking | Writing refinement |

---

## 2. AI Assistance Details

### 2.1 Mathematical Modeling Assistance

| Task | AI Tool | Specific Use | Team Verification |
|------|---------|--------------|-------------------|
| Literature Review | ChatGPT | Summarizing battery modeling approaches and identifying relevant papers | Team verified all sources and read original papers |
| Equation Formulation | ChatGPT, Claude | Suggesting ODE structures for SOC modeling | Team derived and validated all equations independently |
| Parameter Research | ChatGPT | Finding typical Li-ion battery parameters from literature | All parameters verified against primary sources |

### 2.2 Code Implementation Assistance

| Task | AI Tool | Specific Use | Team Verification |
|------|---------|--------------|-------------------|
| ODE Solver | ChatGPT, Copilot | Implementing scipy.integrate.solve_ivp with event detection | Tested against analytical solutions |
| Data Analysis | ChatGPT, Copilot | Pandas operations for dataset processing | Results manually verified |
| Visualization | Copilot | Matplotlib plotting code for figures | All figures reviewed for accuracy |
| Debugging | ChatGPT | Identifying and fixing runtime errors | All fixes tested before acceptance |

### 2.3 Writing Assistance

| Task | AI Tool | Specific Use | Team Verification |
|------|---------|--------------|-------------------|
| Draft Refinement | ChatGPT | Improving clarity and flow of technical writing | All content reviewed and modified by team |
| Grammar Check | Grammarly | Correcting grammatical errors | Team accepted/rejected suggestions |
| LaTeX Formatting | ChatGPT | Equation formatting and document structure | All formatting verified |

### 2.4 Specific AI Interactions

**Literature Research**
- **Query**: "What are the main approaches for modeling lithium-ion battery state of charge?"
- **AI Response**: Provided overview of equivalent circuit models, electrochemical models, and data-driven approaches
- **Team Action**: Used as starting point for literature review; read and cited primary sources directly

**Model Development**
- **Query**: "How to model temperature effects on Li-ion battery capacity?"
- **AI Response**: Suggested Arrhenius-type relationship and typical parameters
- **Team Action**: Verified against NASA dataset and published studies; derived our own temperature model

**Code Implementation**
- **Query**: "Implement RK45 ODE solver for battery discharge model"
- **AI Response**: Provided template code using scipy
- **Team Action**: Adapted to our specific model equations, added event detection for shutdown threshold

**Writing Refinement**
- **Query**: Submitted draft paragraphs for clarity improvement
- **AI Response**: Suggested restructured sentences and clearer phrasing
- **Team Action**: Reviewed all suggestions, accepted improvements, rejected changes to technical meaning

---

## 3. Verification Process

All AI-assisted content underwent rigorous verification:

| Verification Method | Description | Status |
|---------------------|-------------|--------|
| Source Verification | All literature references checked against original papers | Completed |
| Equation Derivation | All model equations independently derived by team | Completed |
| Code Testing | Unit tests and integration tests for all code | Completed |
| Numerical Validation | Model outputs compared to dataset values | Completed |
| Writing Review | All text reviewed by at least two team members | Completed |

---

## 4. Summary of AI Contribution

| Component | AI Contribution | Team Contribution | Notes |
|-----------|-----------------|-------------------|-------|
| Literature Review | ~30% (initial search) | ~70% (reading, analysis) | AI helped identify papers; team read and analyzed |
| Mathematical Model | ~15% (suggestions) | ~85% (derivation, validation) | Core concepts by team; AI assisted formulation |
| Data Analysis | ~20% (code assistance) | ~80% (methodology, interpretation) | Team designed analysis; AI helped implement |
| Code Implementation | ~40% (syntax, templates) | ~60% (logic, debugging) | AI provided templates; team adapted and tested |
| Paper Writing | ~25% (drafts, grammar) | ~75% (content, structure) | AI refined drafts; team wrote all content |
| Figures | ~15% (plotting code) | ~85% (design, content) | AI helped with matplotlib; team designed visuals |

**Overall Estimated AI Contribution: ~25%**

AI tools provided substantial assistance throughout the project, particularly in code implementation and writing refinement. However, all core intellectual contributions - including the mathematical modeling approach, data analysis methodology, and key insights - were developed by our team. All AI-generated content was thoroughly reviewed and validated before inclusion.

---

## 5. Compliance Statement

We certify that our use of AI tools complies with COMAP's MCM/ICM guidelines:

1. **Disclosure**: All AI tools and their usage are fully disclosed in this report
2. **Verification**: All AI-generated content has been verified by team members
3. **Originality**: Core modeling concepts and methodology are original team work
4. **Responsibility**: Team takes full responsibility for all content in our submission

---

## Certification

We certify that this AI Use Report accurately represents our use of AI tools. While AI provided valuable assistance, the core intellectual contributions are our original work. All AI-generated content has been reviewed and validated by our team, and we accept full responsibility for all content in our MCM submission.

**Team Control Number**: XXXXXX

**Date**: January 2026

---

*This report was prepared in accordance with the Mathematical Contest in Modeling (MCM) AI usage guidelines.*
