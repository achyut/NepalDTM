#!/bin/zsh
# Upload finished Bagmati G tiles to R2 bucket nepaldtm-g-1m (bagmati/tiles/), and keep syncing new ones.
# A tile is uploaded once; done names are kept in uploaded.txt. Log: fusion/data/gen/r2_sync.log
cd /Users/paudea2/workspace/NepalDTM
T=/Users/paudea2/testground/3dnepal/fusion/data/gen/region/bagmati/tiles
DONE=/Users/paudea2/testground/3dnepal/fusion/data/gen/region/bagmati/uploaded.txt
touch $DONE
while true; do
  for f in $T/G_*.tif(N); do
    n=${f:t}
    [[ $n == *.part.tif ]] && continue
    grep -qx $n $DONE && continue
    if npx wrangler r2 object put "nepaldtm-g-1m/bagmati/tiles/$n" --file $f --remote 2>&1 | grep -q "Upload complete"; then
      echo $n >> $DONE; echo "$(date '+%F %T') uploaded $n"
    else
      echo "$(date '+%F %T') FAILED $n (retry next round)"
    fi
  done
  pgrep -f "g_region.py" >/dev/null || { echo "generator stopped; final pass done $(date)"; break; }
  sleep 300
done
