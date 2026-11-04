"""ro5-items: the display-name column of game/data/items.py and its welcome.items field."""
from game.data import items as item_data


def test_display_names_capitalised():
    names = dict(zip(item_data.NAMES, item_data.DISPLAY_NAMES))
    assert names["gildscarab card"] == "Gildscarab Card"
    assert names["red potion"] == "Red Potion"
    assert names["bill of birds"] == "Bill of Birds"
    assert names["clackbone guard card"] == "Clackbone Guard Card"
    assert item_data.DISPLAY_NAMES[0] == ""


def test_every_real_item_has_a_display_name():
    for name, disp, row in zip(item_data.NAMES, item_data.DISPLAY_NAMES, item_data.ROWS):
        if row[1] != item_data.KIND_NONE:
            assert disp and disp.lower() == name and disp[0].isupper()
        else:
            assert disp == ""


def test_welcome_items_carry_display():
    table = item_data.table_for_client()
    assert len(table) == item_data.N_ITEMS
    assert all(d["display"] == item_data.DISPLAY_NAMES[i] for i, d in enumerate(table))
    assert table[3]["display"] == "Sword" and table[3]["icon"] == "item.sword"
