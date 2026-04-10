import unittest
from producer.order_producer import get_order

class TestOrderProducer(unittest.TestCase):
    def test_get_order_fields(self):
        """Verify that the generated order has all required fields"""
        order = get_order()
        required_fields = [
            "order_id", "user_id", "product_id", "product_name", 
            "category", "amount", "quantity", "payment_method", 
            "ip_address", "timestamp"
        ]
        for field in required_fields:
            self.assertIn(field, order, f"Field {field} missing in generated order")

    def test_order_amount_logic(self):
        """Check if amount is within expected ranges"""
        order = get_order()
        self.assertGreater(order["amount"], 0)
        self.assertIsInstance(order["amount"], float)

    def test_order_types(self):
        """Verify data types of fields"""
        order = get_order()
        self.assertIsInstance(order["quantity"], int)
        self.assertIsInstance(order["category"], str)

if __name__ == "__main__":
    unittest.main()
