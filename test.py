import re

text = """
| Item Number | Date     | Description                  | Amount |
|-------------|----------|------------------------------|--------|
| 1111/7333   | 5/24/2023| CHECK PAID TWICE BY BANK     | 300.00 |

Outstanding Suspense Items: 300.00
### Cleared Checks/Vouchers

| Document Number | Document Date | Document Description                | Document Amount | Payee                                      |
|-----------------|---------------|-------------------------------------|-----------------|--------------------------------------------|
| 6858            | 4/21/2023     | System Generated Check/Voucher      | 300.00          | HILL, DARIAN                               |
| 7104            | 4/21/2023     | System Generated Check/Voucher      | 300.00          | FLOYD, RYAN MATTHEW                        |
{'reconciliation_date': '6/30/2023', 'uncleared': 'checks/vouchers'}
"""

# match = re.search(r"\*\*Reconciliation Date:\*\*\s*(\d{1,2}/\d{1,2}/\d{4})", text)
match = re.search(r"Cleared Checks/Vouchers", text)

print("Match", match)