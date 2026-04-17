# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 1.0.x   | ✅ Yes             |
| < 1.0   | ❌ No              |

## Reporting a Vulnerability

We take security issues seriously. If you discover a security vulnerability, please follow these steps:

1. **DO NOT** open a public issue.
2. Email the maintainer directly at the email associated with the repository.
3. Include the following information:
   - Description of the vulnerability
   - Steps to reproduce (if applicable)
   - Potential impact assessment

We will acknowledge receipt within 48 hours and provide a detailed response within 7 days.

## Security Best Practices for Contributors

- Never commit API keys, tokens, or credentials to the repository.
- Use `pip-audit` to scan dependencies for known vulnerabilities (runs in CI).
- Ensure all user input is validated before being passed to kernel operations.
- Follow the [Spec-Driven Development (SDD)](AGENTS.md) workflow to maintain code quality.

## Dependency Security

This project uses:
- `torch` >= 2.0.0
- `triton` >= 2.1.0

Always use the latest stable versions to benefit from security patches.
