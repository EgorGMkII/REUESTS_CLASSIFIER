from src.classifiers.import_themes import (
    build_paths,
    build_theme_nodes,
    build_tree,
    extract_code_name_pairs,
    mark_leaf_nodes,
    remove_methodical_heading,
    write_json,
)


TEXT = """
МЕТОДИЧЕСКИЙ ПЕРЕЧЕНЬ
0000.0000.0000.1149 Не брать
ТЕМАТИЧЕСКИЙ КЛАССИФИКАТОР ОБРАЩЕНИЙ ГРАЖДАН
0005.0000.0000.0000 Жилищно-коммунальная сфера
0005.0005.0000.0000 Жилище
0005.0005.0056.0000 Коммунальное хозяйство
0005.0005.0056.1149
Оплата жилищно-коммунальных
услуг
0005.0005.0056.1149 Дубликат
0005.0005.0056.1149.0001 Детализация
"""


def test_builds_hierarchy_and_four_block_leaf():
    rows = extract_code_name_pairs(TEXT)
    nodes = mark_leaf_nodes(build_paths(build_theme_nodes(rows)), "four-block-only")
    by_code = {node.code: node for node in nodes}
    leaf = by_code["0005.0005.0056.1149"]
    assert leaf.isLeaf
    assert leaf.sectionName == "Жилищно-коммунальная сфера"
    assert leaf.parentName == "Коммунальное хозяйство"
    assert leaf.pathCodes[-1] == leaf.code
    assert build_tree(nodes)[0]["children"]


def test_five_block_not_api_leaf():
    nodes = mark_leaf_nodes(
        build_paths(build_theme_nodes(extract_code_name_pairs(TEXT))),
        "four-block-only",
    )
    api_leaves = [node for node in nodes if node.isLeaf and node.code.count(".") == 3]
    assert all(node.code.count(".") == 3 for node in api_leaves)


def test_tree_can_be_written():
    output = Path("tree.json")
    with patch.object(Path, "mkdir"), patch.object(Path, "write_text") as write:
        write_json(output, [{"code": "0005.0000.0000.0000", "children": []}])
    assert '"children": []' in write.call_args.args[0]


def test_methodical_heading_is_removed_from_name():
    raw = (
        "Жилищно-коммунальная сфера "
        "0000.ХХХХ.0000.0000 – ПЕРЕЧЕНЬ ТЕМАТИК ТИПОВОГО КЛАССИФИКАТОРА"
    )
    assert remove_methodical_heading(raw) == "Жилищно-коммунальная сфера"
from pathlib import Path
from unittest.mock import patch
