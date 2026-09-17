from research_program.triadic_multi_legal_direction_probe import design


def test_probe_support_and_neighbor_rule():
    needs, neighbors = design.probe_needs()
    assert len(needs) == 1404
    assert len(neighbors) == 1404 * 3
    for row in needs[::137]:
        for actor in range(3):
            donor = neighbors[row, actor]
            assert donor != row
            assert sum(a != b for a, b in zip(row, donor)) == 1
            assert all(row[i] == donor[i] for i in range(3) if i != actor)


def test_case_expansion():
    spec = design.source_partitions()
    cases = design.make_cases(spec)
    assert cases['case_count'] == 1404 * 6 * 6 * 3
    assert cases['case_order'] == 'need,layout,owner,sender'
    assert set(cases['sender']) == {0, 1, 2}
