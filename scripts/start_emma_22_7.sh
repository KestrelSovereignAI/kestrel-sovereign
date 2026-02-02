#!/bin/bash
#
# Start Emma 22/7 Scheduler
#
# Runs Emma in autonomous 22/7 mode:
# - 22 hours of work (training, tasks)
# - 2 hours of sleep (consolidation, reflection)
#
# Usage:
#   ./scripts/start_emma_22_7.sh                    # Default (sleep 00:00-02:00)
#   ./scripts/start_emma_22_7.sh --sleep-start 3    # Custom sleep window
#   ./scripts/start_emma_22_7.sh --background       # Run in background
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# Default values
DB_PATH="agent_data/kestrel_prime.db"
SLEEP_START=0
SLEEP_END=2
BACKGROUND=false
VERBOSE=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --db)
            DB_PATH="$2"
            shift 2
            ;;
        --sleep-start)
            SLEEP_START="$2"
            shift 2
            ;;
        --sleep-end)
            SLEEP_END="$2"
            shift 2
            ;;
        --background|-b)
            BACKGROUND=true
            shift
            ;;
        --verbose|-v)
            VERBOSE=true
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --db PATH           Path to Emma's database (default: agent_data/kestrel_prime.db)"
            echo "  --sleep-start HOUR  Hour to start sleep (0-23, default: 0)"
            echo "  --sleep-end HOUR    Hour to end sleep (0-23, default: 2)"
            echo "  --background, -b    Run in background"
            echo "  --verbose, -v       Enable verbose logging"
            echo "  --help, -h          Show this help"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Verify database exists
if [ ! -f "$DB_PATH" ] && [ ! -d "$DB_PATH" ]; then
    echo "Error: Database not found at $DB_PATH"
    echo "Run inception_service.py first to create Emma."
    exit 1
fi

# Verify KESTREL_DATA_KEY is set
if [ -z "$KESTREL_DATA_KEY" ]; then
    if [ -f ".env" ]; then
        # Try to load from .env
        export $(grep KESTREL_DATA_KEY .env | xargs)
    fi
    if [ -z "$KESTREL_DATA_KEY" ]; then
        echo "Warning: KESTREL_DATA_KEY not set. Some features may not work."
    fi
fi

# Create logs directory
mkdir -p logs

# Build command
CMD="uv run python scripts/emma_scheduler.py $DB_PATH --sleep-start $SLEEP_START --sleep-end $SLEEP_END"
if [ "$VERBOSE" = true ]; then
    CMD="$CMD --verbose"
fi

echo "=============================================="
echo "  EMMA 22/7 SCHEDULER"
echo "=============================================="
echo ""
echo "  Database:     $DB_PATH"
echo "  Sleep Window: ${SLEEP_START}:00 - ${SLEEP_END}:00"
echo "  Mode:         $(if [ "$BACKGROUND" = true ]; then echo "Background"; else echo "Foreground"; fi)"
echo ""

if [ "$BACKGROUND" = true ]; then
    LOG_FILE="logs/emma_scheduler.log"
    PID_FILE=".emma_scheduler.pid"

    # Check if already running
    if [ -f "$PID_FILE" ]; then
        OLD_PID=$(cat "$PID_FILE")
        if kill -0 "$OLD_PID" 2>/dev/null; then
            echo "Error: Scheduler already running (PID: $OLD_PID)"
            echo "Run: kill $OLD_PID to stop it first"
            exit 1
        fi
    fi

    echo "Starting in background..."
    echo "  Log: $LOG_FILE"
    echo "  PID: $PID_FILE"
    echo ""

    nohup $CMD >> "$LOG_FILE" 2>&1 &
    echo $! > "$PID_FILE"

    echo "Started! PID: $(cat $PID_FILE)"
    echo ""
    echo "Commands:"
    echo "  Monitor: tail -f $LOG_FILE"
    echo "  Stop:    kill \$(cat $PID_FILE)"
else
    echo "Starting in foreground (Ctrl+C to stop)..."
    echo ""
    $CMD
fi
