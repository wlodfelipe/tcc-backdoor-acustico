#!/usr/bin/env bash
# Wait until every given kernel reaches a terminal state, then print the final statuses.
# Only the CLI's status line counts: network or auth errors are retried, never read as "finished".
#   scripts/kaggle_wait.sh tcc-principal-a tcc-principal-c
K=~/.local/bin/kaggle
pending=("$@")
while [ ${#pending[@]} -gt 0 ]; do
  still=()
  for k in "${pending[@]}"; do
    s=$($K kernels status "flpwld/$k" 2>&1)
    if [[ $s =~ has\ status\ \"KernelWorkerStatus\.(COMPLETE|ERROR|CANCEL_ACKNOWLEDGED|CANCEL_REQUESTED)\" ]]; then
      echo "$s"
    else
      [[ $s =~ has\ status ]] || echo "retrying $k: ${s%%$'\n'*}" >&2
      still+=("$k")
    fi
  done
  pending=("${still[@]}")
  [ ${#pending[@]} -gt 0 ] && sleep 120
done
exit 0
