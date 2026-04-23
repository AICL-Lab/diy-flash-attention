#!/bin/bash
# Sync OpenSpec archive to changelog
# This script creates changelog entries from archived OpenSpec proposals

set -e

ARCHIVE_DIR="openspec/changes/archive"
CHANGELOG_DIR="changelog/archive"

echo "Syncing OpenSpec archive to changelog..."

# Check if archive directory exists
if [ ! -d "$ARCHIVE_DIR" ]; then
    echo "No archive directory found at $ARCHIVE_DIR"
    exit 0
fi

# Ensure changelog archive directory exists
mkdir -p "$CHANGELOG_DIR"

# Process each archived proposal
for dir in "$ARCHIVE_DIR"/*/; do
    # Skip if no directories found
    [ -d "$dir" ] || continue

    if [ -f "${dir}proposal.md" ]; then
        basename=$(basename "$dir")

        # Extract date from directory name (expected format: YYYY-MM-DD-title)
        date_part=$(echo "$basename" | cut -d'-' -f1-3)

        # Extract title from proposal
        title=$(grep -m1 "^# " "${dir}proposal.md" | sed 's/^# //' || echo "Untitled Change")

        changelog_file="$CHANGELOG_DIR/${basename}.md"

        # Skip if changelog entry already exists
        if [ -f "$changelog_file" ]; then
            echo "Skipping $basename - changelog entry already exists"
            continue
        fi

        echo "Creating changelog entry: $changelog_file"

        # Extract summary from proposal if available
        summary=""
        if grep -q "^## Summary" "${dir}proposal.md"; then
            summary=$(sed -n '/^## Summary/,/^## /p' "${dir}proposal.md" | head -n -1 | tail -n +2 | sed 's/^ *//')
        fi

        # Extract tasks from proposal if available
        tasks=""
        if grep -q "^## Tasks" "${dir}proposal.md"; then
            tasks=$(sed -n '/^## Tasks/,/^## /p' "${dir}proposal.md" | head -n -1 | tail -n +2)
        fi

        # Create changelog entry
        cat > "$changelog_file" << EOF
# ${title}

Date: ${date_part}

## Summary
${summary:-See proposal for details.}

## Changes
${tasks:-See proposal for details.}

## Related
- OpenSpec Proposal: openspec/changes/archive/${basename}/
EOF

        echo "Created: $changelog_file"
    fi
done

echo ""
echo "Sync complete!"
