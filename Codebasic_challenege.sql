/* Question 1 */
Select Distinct market
from dim_customer
where customer = 'Atliq Exclusive' and region = 'APAC'
order by 1 asc;

/* Question 2*/
Select *, Round (100 * ((unique_product_code_2021 - unique_product_code_2020)/unique_product_code_2020), 2) as percentage_change
From (
SELECT 
  COUNT(DISTINCT CASE WHEN fiscal_year = 2020 THEN product_code END) AS unique_product_code_2020, 
  COUNT(DISTINCT CASE WHEN fiscal_year = 2021 THEN product_code END) AS unique_product_code_2021 
FROM fact_sales_monthly
) as unique_codes;

/* Question 3*/
select segment, count( distinct product) as unique_prod_count
from dim_product
group by 1
order by 2 desc;

/* Question 4*/
with unique_products as (
select segment, count( distinct case when fiscal_year = 2020 then product end )as unique_product_2020, 
count( distinct case when fiscal_year = 2021 then product end )as unique_product_2021
from dim_product as dp right join fact_sales_monthly as fsm
on dp.product_code = fsm.product_code
group by 1)
Select *, (unique_product_2021 - unique_product_2020) as difference 
from unique_products
order by 4 desc ;

/* Question 5*/
select fmc.product_code, product, manufacturing_cost
from fact_manufacturing_cost as fmc inner join dim_product as dp
on fmc.product_code = dp.product_code
where manufacturing_cost = (select min(manufacturing_cost) from fact_manufacturing_cost) or 
manufacturing_cost = (select max(manufacturing_cost) from fact_manufacturing_cost)
;

/* Question 6*/
select fpi.customer_code, customer,pre_invoice_discount_pct, Round((select avg(pre_invoice_discount_pct) from fact_pre_invoice_deductions),2) as avg_discount
from fact_pre_invoice_deductions as fpi inner join dim_customer as dc
on fpi.customer_code = dc.customer_code
where pre_invoice_discount_pct > (select avg(pre_invoice_discount_pct) from fact_pre_invoice_deductions) and fiscal_year = 2021
order by 3 desc
limit 5;

/* Question 7*/
with Atliq_cust as (
select extract(month from date) as month,fiscal_year, product_code, sold_quantity
from fact_sales_monthly as fsm inner join dim_customer as dc
on fsm.customer_code = dc.customer_code
where customer = 'Atliq Exclusive'
)

select month, ac.fiscal_year, Round((sold_quantity * gross_price), 2) as gross_sales
from Atliq_cust as ac inner join fact_gross_price as fgp
on ac.product_code = fgp.product_code and ac.fiscal_year = fgp.fiscal_year
order by 1 asc, 2 asc, 3 desc;

/* Question 8*/
Select quarter, sum(sold_quantity) as total_sold_quantity
from (
Select *, quarter(date) as quarter
from fact_sales_monthly
) as fsm
group by 1
order by 2 desc;

/* Question 9*/
with Atliq_cust as (
select extract(month from date) as month,fiscal_year, product_code, channel, sold_quantity
from fact_sales_monthly as fsm inner join dim_customer as dc
on fsm.customer_code = dc.customer_code
)
, gross_sales_total as (
select channel, month, ac.fiscal_year, Round((sold_quantity * gross_price), 2) as gross_sales
from Atliq_cust as ac inner join fact_gross_price as fgp
on ac.product_code = fgp.product_code and ac.fiscal_year = fgp.fiscal_year
where ac.fiscal_year = 2021)
, sales_by_channel as (
select channel, sum(gross_sales) as gross_sales_mln, (select sum(gross_sales) from gross_sales_total) as total_sales
from gross_sales_total
group by 1)

select channel, gross_sales_mln, Round((((total_sales - gross_sales_mln)/total_sales) * 100.0),2) as percentage
from sales_by_channel
order by 3 desc;

/* Question 10*/
select division, product_code, product, rank_order
from (
select division, dp.product_code as product_code, product, sum(sold_quantity) as total_sold_quantity, 
row_number () over (partition by division order by sum(sold_quantity) desc) as rank_order
from dim_product as dp inner join fact_sales_monthly as fsm
on dp.product_code = fsm.product_code
group by 1,2,3 ) as col 
where rank_order < 4;
