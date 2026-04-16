# Changelog Directory

This directory contains the project's changelog documentation.

## Structure

```
changelog/
├── README.md                  # This file
├── CHANGELOG.md              # Main changelog (English) - Keep a Changelog format
├── CHANGELOG.zh-CN.md        # Chinese changelog - 中文更新日志
└── archive/                  # Archived detailed changelogs
    └── *.md                  # Individual changelog files by release
```

## Files

| File | Description | Language |
|------|-------------|----------|
| [CHANGELOG.md](./CHANGELOG.md) | Main changelog following Keep a Changelog format | English |
| [CHANGELOG.zh-CN.md](./CHANGELOG.zh-CN.md) | Chinese version of changelog | 中文 |

## Archives

The `archive/` directory contains detailed changelogs for each release, organized by date and version.

## Versioning

This project follows [Semantic Versioning](https://semver.org/):

- **MAJOR**: Incompatible API changes
- **MINOR**: Backwards-compatible new features
- **PATCH**: Backwards-compatible bug fixes

## Viewing Changelogs

- **Latest Changes**: See [CHANGELOG.md](./CHANGELOG.md) or [CHANGELOG.zh-CN.md](./CHANGELOG.zh-CN.md)
- **Detailed History**: Browse the [archive/](./archive/) directory
- **Git History**: Use `git log` for commit-level changes
