# Physical AI & Humanoid Robotics Constitution

## Core Principles

### I. Education-First
Content and features must prioritize educational value above all else. Every addition to the platform should directly contribute to learning outcomes for students of physical AI and humanoid robotics. Educational content must be accurate, accessible, and practically applicable.

### II. Accessibility & Inclusion (NON-NEGOTIABLE)
All content and features must be accessible to diverse audiences including those with disabilities and different language backgrounds. Urdu localization is not optional but required. WCAG 2.1 AA compliance is mandatory for all UI components.

### III. Test-First (NON-NEGOTIABLE)
TDD mandatory: Tests written → User approved → Tests fail → Then implement; Red-Green-Refactor cycle strictly enforced. All functionality must have corresponding tests before deployment.

### IV. Performance-Driven
All features must meet defined performance benchmarks: Page load time under 3 seconds, chatbot response time under 2 seconds, and 99.9% uptime. Performance regressions are not acceptable.

### V. Reproducibility & Documentation
All examples, tutorials, and processes must be fully reproducible with clear documentation. Code examples must include expected outputs and environment setup instructions. Version control and changelog maintenance are required.

### VI. Spec-Driven Development
All implementation must strictly follow the defined specifications. Changes to implementation must be reflected in the spec first before implementation. No feature creep is allowed without spec updates.

## Additional Constraints

### Technology Stack Requirements
- Docusaurus for documentation platform
- FastAPI for backend services
- Neon DB for structured data
- Qdrant for vector storage
- OpenAI API for AI services
- GitHub Pages for deployment

### Security & Compliance Standards
- GDPR compliance for user data
- WCAG 2.1 AA accessibility compliance
- Secure authentication and data handling
- Regular security reviews and penetration testing

## Development Workflow

### Code Review Requirements
- All PRs must include test coverage
- Educational content accuracy verification
- Accessibility compliance check
- Performance impact assessment

### Quality Gates
- All tests must pass
- Performance benchmarks met
- Accessibility validation passed
- Educational content reviewed by domain expert

## Governance

Constitution supersedes all other practices; Amendments require documentation, approval, and migration plan. All PRs/reviews must verify compliance; Complexity must be justified; Implementation must align with spec-driven approach.

**Version**: 1.0.0 | **Ratified**: 2025-12-27 | **Last Amended**: 2025-12-27
