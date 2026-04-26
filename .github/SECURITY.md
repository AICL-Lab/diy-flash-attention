# Security Policy

## Supported Versions

This repository is maintained as an archive-ready educational project. The default branch is the only supported line for security fixes; older tags and historical snapshots should be treated as read-only reference material.

## Reporting a Vulnerability

If you discover a security issue:

1. **Do not** open a public issue with exploit details.
2. Use GitHub Security Advisories if available, or contact the maintainer privately via the address associated with the repository.
3. Include:
   - a clear description of the issue
   - the affected file, command, or workflow
   - reproduction steps or a minimal proof of concept
   - impact and any suggested mitigation

Response times are best-effort; this repository is maintained lightly, but credible security reports will be reviewed and triaged.

## Contributor Guidance

- Never commit API keys, tokens, or credentials to the repository.
- Keep dependencies and version requirements aligned across config files, CI, and docs.
- Follow the OpenSpec-first workflow described in [`AGENTS.md`](../AGENTS.md) when changing behavior or workflows.
- Report unsafe defaults in docs, examples, or automation just as you would report code defects.

## Dependency Security

The project depends primarily on `torch` and `triton`. If a dependency-level vulnerability affects this repository's documented setup or workflows, update the relevant config, docs, and CI guidance together.
