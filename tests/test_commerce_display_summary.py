import unittest
from commerce import commerce_summary

class CommerceDisplaySummaryTests(unittest.TestCase):
    def test_names_sources_and_unknown_amounts_come_from_confirmed_records(self):
        result=commerce_summary({"purchases":[
            {"id":1,"status":"confirmed","product_title":"Повний пакет","currency":"XTR","amount":250,"source":"telegram"},
            {"id":2,"status":"confirmed","product_title":"Кейси","currency":"XTR","amount":None,"source":"legacy_telegram"},
            {"id":3,"status":"refunded","product_title":"Повернено","currency":"UAH","amount":125,"source":"admin_external"}],
            "manual_grants":[{"product_title":"Митні компетенції"}],
            "legacy_access":[{"product_title":"Старий пакет"}]})
        self.assertEqual(["telegram","legacy_telegram"],result["purchase_sources"])
        self.assertEqual(["Митні компетенції"],result["manual_titles"])
        self.assertEqual(["Старий пакет"],result["legacy_titles"])
        self.assertEqual(1,result["unknown_amount_count"])
        self.assertEqual({"XTR":250,"UAH":0},result["payment_totals"])
