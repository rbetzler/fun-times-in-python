from datetime import datetime
from typing import List

from finance.trading import utils as trading_utils
from finance.utilities import utils


class TDOrderExecutor:
    def __init__(
            self,
            environment: str,
            model_id: str,
    ):
        self.environment = environment
        self.model_id = model_id

    def get_new_orders(self) -> List[trading_utils.TDOrder]:
        """Get new orders from the latest model run; then, convert df to TDOrder"""
        query = f'''
        with
        orders as (
          select
              model_id
            , direction
            , asset
            , denormalized_prediction as price
            , quantity
            , symbol
            , dense_rank() over (order by model_datetime desc) as dr
          from {self.environment}.decisions
          where model_id = '{self.model_id}'
            and model_datetime::date = current_date
        )
        select distinct *
        from orders
        where dr = 1
        '''
        df = utils.query_db(query=query)
        orders = []
        for row in df.itertuples():
            order = trading_utils.TDOrder(
                symbol=row.symbol,
                instruction=row.direction,
                price=row.price,
                quantity=row.quantity,
            )
            orders.append(order)
        return orders

    def execute(self):
        """Get new orders, open orders and existing positions. Determine and place new trades."""
        print(f'Getting new orders: {datetime.utcnow()}')
        new_orders = self.get_new_orders()

        print(f'Getting existings positions and open orders: {datetime.utcnow()}')
        existing_positions = {p.symbol for p in trading_utils.TDAccounts().get_positions()}
        open_orders = {o.symbol for o in trading_utils.get_orders()}
        symbols_to_ignore = existing_positions.union(open_orders)

        print(f'Netting out existing positions or open orders: {datetime.utcnow()}')
        orders_to_place = []
        for new_order in new_orders:
            if new_order.symbol not in symbols_to_ignore:
                orders_to_place.append(new_order)

        print(f'Placing net new orders: {datetime.utcnow()}')
        # for order in orders_to_place:
        #     trading_utils.place_order(order=order)


if __name__ == '__main__':
    TDOrderExecutor(environment='dev', model_id='s0').execute()
