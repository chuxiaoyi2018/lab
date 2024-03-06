# 批量修改命名
for f in qwen_block*; do mv "$f" "${f/#qwen_block/block}"; done

