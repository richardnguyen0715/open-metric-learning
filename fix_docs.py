import pathlib

p = pathlib.Path('oml/functional/metrics.py')
content = p.read_text()

content = content.replace('''        gt_ids: Gallery indices relevant to every query with the size of ``n_query``.
            Every element is within the range ``(0, n_gallery - 1)``
        query_categories:''', '''        gt_ids: Gallery indices relevant to every query with the size of ``n_query``.
            Every element is within the range ``(0, n_gallery - 1)``
        distances: Distances between queries and retrieved items.
        query_categories:''')

content = content.replace('''        map_top_k: Values of ``k`` to calculate ``map@k`` (`Mean Average Precision`)
        reduce:''', '''        map_top_k: Values of ``k`` to calculate ``map@k`` (`Mean Average Precision`)
        ndcg_top_k: Values of ``k`` to calculate ``ndcg@k`` (`Normalized Discounted Cumulative Gain`)
        calc_global_pr_auc_metric: Whether to compute global PR-AUC
        reduce:''')
p.write_text(content)
