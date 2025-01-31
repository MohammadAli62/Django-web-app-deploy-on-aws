from django.shortcuts import render,redirect, get_object_or_404
from .models import Customer, UploadedFileName
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import io
from io import StringIO
import pandas as pd
import urllib, base64
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.io as pio
import csv

max_date =  "2024-01-10T12:00"
selection_value = 2
customer_to_show = 40
df = None
uploaded_file_name = None

def home(request):
    return render(request, "customer/home.html")


def customer_retail_data_showing(request):

    global df, uploaded_file_name
    if request.method == 'POST':
        csv_file = request.FILES['dataset']

        if not csv_file.name.endswith('.csv'):
            # Handle invalid file type
            return render(request, 'customer/retail_data.html', {'message': 'This is not a CSV file, please upload the CSV file'})

        uploaded_file_name = csv_file.name

        # Process the CSV file
        decoded_file = csv_file.read().decode('unicode_escape').splitlines()
        reader = csv.reader(decoded_file)
        data = []
        
        for row in reader:
            data.append(row)

        dataset = pd.DataFrame(data)
        # Setting the second row as the header
        dataset.columns = dataset.iloc[0]
        dataset = dataset[1:]
        df = dataset

        df_html = df.to_html(index=False, classes='table table-bordered table-striped table-hover', max_rows=35)
        return render(request, 'customer/retail_data.html', {"df":df_html, "uploaded_file_name":uploaded_file_name, "title":"Uploaded Dataset"})
    else:
        if df is not None:
            #df = pd.read_csv("D:\CRM Project\CRM\customer\Cleaned_OnlineRetail.csv", encoding='unicode_escape', nrows = 35, index_col=False)
            df_html = df.to_html(index=False, classes='table table-bordered table-striped table-hover', max_rows=35)
            return render(request, "customer/retail_data.html",{"df":df_html,"uploaded_file_name":uploaded_file_name,"title":"Retail Dataset"})
        else:
            return render(request, "customer/retail_data.html",{"message":"Please upload the dataset first.....","title":"Upload Dataset"})



def cutomize_segmentation(request):
    
    global max_date, selection_value, customer_to_show, df, uploaded_file_name

    if request.method == "POST":
        max_date = request.POST.get('datetime')
        #print(max_date)
        selection_value = int(request.POST.get('selection'))
        customer_to_show = request.POST.get('customer_to_show')
        #print(to_date,selection_value)
        #df = pd.read_csv("D:\CRM Project\CRM\customer\Cleaned_OnlineRetail.csv", encoding='unicode_escape', index_col=False)
        max_date_original = df['InvoiceDate'].max()
        no_of_rows_in_data_set = df.shape[0]
        arr = df.CustomerID.unique()
        no_of_customers = len(arr)
        return render(request, "customer/segmentation.html", {"uploaded_file_name":uploaded_file_name, "message": "Successfuly Updated", "customer_to_show":customer_to_show, "selection_value":selection_value, "max_date":max_date, "max_date_original":max_date_original, "no_of_rows_in_data_set":no_of_rows_in_data_set, "no_of_customers":no_of_customers})
    
    else:
        if df is not None:
            #df = pd.read_csv("D:\CRM Project\CRM\customer\Cleaned_OnlineRetail.csv", encoding='unicode_escape', index_col=False)
            max_date_original = df['InvoiceDate'].max()
            no_of_rows_in_data_set = df.shape[0]
            arr = df.CustomerID.unique()
            no_of_customers = len(arr)
            return render(request, "customer/segmentation.html",{"uploaded_file_name":uploaded_file_name, "max_date_original":max_date_original, "max_date":max_date, "no_of_rows_in_data_set":no_of_rows_in_data_set, "no_of_customers":no_of_customers, "customer_to_show":customer_to_show})
        else:
            return redirect("retail_data")



def func_choise_2(row, good_one, avg_one):

    if row["Clusters"] ==  good_one:
        return 'Whales'
    elif row["Clusters"] == avg_one:
        return 'Average'
    
def func_choise_3(row, good_one, avg_one):

    if row["Clusters"] ==  good_one:
        return 'Whales'
    elif row["Clusters"] == avg_one:
        return 'Average'
    else:
        return 'Lapsed'


def func_choise_4(row,excellent_one, good_one, avg_one):

    if row["Clusters"] ==  excellent_one:
        return 'Whales'
    elif row["Clusters"] == good_one:
        return 'Shark'
    elif row["Clusters"] == avg_one:
        return 'Average'  
    else:
        return 'Lapsed'


    if selection_value == 2:
        df_average = df[df['Group'] == 'Average']
        df_whales = df[df['Group'] == 'Whales']
        
        return df_average,df_whales

    elif selection_value == 3:
        df_lapsed = df[df['Group'] == 'Lapsed']
        df_average = df[df['Group'] == 'Average']
        df_whales = df[df['Group'] == 'Whales']

        return df_lapsed,df_average,df_whales

    elif selection_value == 4:
        df_lapsed = df[df['Group'] == 'Lapsed']
        df_average = df[df['Group'] == 'Average']
        df_shark = df[df['Group'] == 'Shark']
        df_whales = df[df['Group'] == 'Whales']

        return df_lapsed,df_average,df_shark,df_whales



def RFM_analyzer(request):
    global max_date , selection_value, customer_to_show, df, uploaded_file_name

    if request.method == "POST":
        try:
            #df = pd.read_csv("D:\CRM Project\CRM\customer\Cleaned_OnlineRetail.csv", encoding='unicode_escape', index_col=False)
            df.dropna(subset=['CustomerID','Quantity','UnitPrice'], inplace=True)
            df.drop_duplicates(subset=['CustomerID','Quantity','UnitPrice','InvoiceDate'], inplace=True)
            #df['CustomerID'] = df['CustomerID'].astype(int)    # this used when we not have any nan value in column
            df['CustomerID'] = pd.to_numeric(df['CustomerID'])
            df['Quantity'] = pd.to_numeric(df['Quantity'])    # this is used to overcome/ignored the nan value when we do conversion of types
            df['UnitPrice'] = pd.to_numeric(df['UnitPrice'])
            df['InvoiceDate'] = pd.to_datetime(df["InvoiceDate"])
            recency = df.groupby(["CustomerID"]).agg({"InvoiceDate": lambda x:((pd.to_datetime(max_date)-x.max()).days)})
            frequency = df.drop_duplicates(subset="InvoiceNo").groupby(["CustomerID"])[["InvoiceNo"]].count()
            df["total"] = df["Quantity"]*df["UnitPrice"]
            monetary = df.groupby(["CustomerID"])[["total"]].sum()
            RFM = pd.concat([recency,frequency,monetary], axis=1)
            RFM["total"] = RFM["total"].round(3)

            RFM.rename(columns={'InvoiceDate': 'Recency', 'InvoiceNo':'Frequency','total':'Monetary'}, inplace=True)

            scaler=StandardScaler()
            scaled = scaler.fit_transform(RFM)

            kmeans = KMeans(n_clusters=selection_value)
            kmeans.fit(scaled)
            RFM["Clusters"] = (kmeans.labels_+1)

            final = RFM.groupby("Clusters")[["Recency","Frequency","Monetary"]].mean()
            final = final.sort_values(by=['Recency', 'Frequency', 'Monetary'], ascending=[True, False, False])

            UploadedFileName.objects.all().delete()

            file_name_saving_to_database = UploadedFileName(file_name=uploaded_file_name, file_identifier=1)
            file_name_saving_to_database.save()

            if selection_value == 2:
                good_one = final.index[0]
                avg_one = final.index[1]
                RFM['Group'] = RFM.apply(func_choise_2, args=(good_one,avg_one), axis=1)

                Customer.objects.all().delete()

                for idx, row in RFM.iterrows():
                    customer = Customer(customer_id=idx, recency=row['Recency'], frequency=row['Frequency'], monetary=row['Monetary'], clusters=row['Clusters'], group=row['Group'])
                    customer.save()

                df_average = RFM[RFM['Group'] == 'Average']
                df_whales = RFM[RFM['Group'] == 'Whales']

                df_average = df_average.head(int(customer_to_show))
                df_whales = df_whales.head(int(customer_to_show))

                df_average = df_average.to_html( classes='table table-bordered table-striped')
                df_whales = df_whales.to_html( classes='table table-bordered table-striped')
                
                return render(request, "customer/retail_data.html", {"file_name":uploaded_file_name, "selection_value":2, "df_average":df_average,"df_whales":df_whales, "title":"Segmented Customer Based"})

            elif selection_value == 3:
                good_one = final.index[0]
                avg_one = final.index[1]
                RFM['Group'] = RFM.apply(func_choise_3, args=(good_one,avg_one), axis=1)

                Customer.objects.all().delete()

                for idx, row in RFM.iterrows():
                    customer = Customer(customer_id=idx, recency=row['Recency'], frequency=row['Frequency'], monetary=row['Monetary'], clusters=row['Clusters'], group=row['Group'])
                    customer.save()

                df_lapsed = RFM[RFM['Group'] == 'Lapsed']
                df_average = RFM[RFM['Group'] == 'Average']
                df_whales = RFM[RFM['Group'] == 'Whales']

                df_lapsed = df_lapsed.head(int(customer_to_show))
                df_average = df_average.head(int(customer_to_show))
                df_whales = df_whales.head(int(customer_to_show))

                df_lapsed = df_lapsed.to_html( classes='table table-bordered table-striped')
                df_average = df_average.to_html( classes='table table-bordered table-striped')
                df_whales = df_whales.to_html( classes='table table-bordered table-striped')

                return render(request, "customer/retail_data.html",{"file_name":uploaded_file_name, "selection_value":3, "df_lapsed":df_lapsed, "df_average":df_average,"df_whales":df_whales, "title":"Segmented Customer Based"})

            elif selection_value == 4:
                excellent_one = final.index[0]
                good_one = final.index[1]
                avg_one = final.index[2]
                RFM['Group'] = RFM.apply(func_choise_4, args=(excellent_one,good_one,avg_one), axis=1)

                Customer.objects.all().delete()

                for idx, row in RFM.iterrows():
                    customer = Customer(customer_id=idx, recency=row['Recency'], frequency=row['Frequency'], monetary=row['Monetary'], clusters=row['Clusters'], group=row['Group'])
                    customer.save()

                df_lapsed = RFM[RFM['Group'] == 'Lapsed']
                df_average = RFM[RFM['Group'] == 'Average']
                df_shark = RFM[RFM['Group'] == 'Shark']
                df_whales = RFM[RFM['Group'] == 'Whales']

                df_lapsed = df_lapsed.head(int(customer_to_show))
                df_average = df_average.head(int(customer_to_show))
                df_shark = df_shark.head(int(customer_to_show))
                df_whales = df_whales.head(int(customer_to_show))

                df_lapsed = df_lapsed.to_html( classes='table table-bordered table-striped')
                df_average = df_average.to_html( classes='table table-bordered table-striped')
                df_shark = df_shark.to_html( classes='table table-bordered table-striped')
                df_whales = df_whales.to_html( classes='table table-bordered table-striped')

                return render(request, "customer/retail_data.html",{"file_name":uploaded_file_name, "selection_value":4, "df_lapsed":df_lapsed, "df_average":df_average,"df_shark":df_shark,"df_whales":df_whales,  "title":"Segmented Customer Based"})
        except KeyError as e:
            return render(request, "customer/error.html", {"message": f"KeyError: {e}"})
        except Exception as e:
            return render(request, "customer/error.html", {"message": f"An error occurred: {e}"})

    else:
    
        customer_queried =  Customer.objects.values("customer_id", "recency", "frequency", "monetary", "clusters", "group")

        file_name_from_database = get_object_or_404(UploadedFileName, file_identifier=1)

        customer_list = list(customer_queried)
        dff = pd.DataFrame(customer_list)

        df_lapsed = dff[dff['group'] == 'Lapsed']
        df_average = dff[dff['group'] == 'Average']
        df_shark = dff[dff['group'] == 'Shark']
        df_whales = dff[dff['group'] == 'Whales']

        df_lapsed = df_lapsed.head(int(customer_to_show))
        df_average = df_average.head(int(customer_to_show))
        df_shark = df_shark.head(int(customer_to_show))
        df_whales = df_whales.head(int(customer_to_show))

        df_lapsed = df_lapsed.to_html( classes='table table-bordered table-striped', index=False)
        df_average = df_average.to_html( classes='table table-bordered table-striped', index=False)
        df_shark = df_shark.to_html( classes='table table-bordered table-striped', index=False)
        df_whales = df_whales.to_html( classes='table table-bordered table-striped', index=False)

        return render(request, "customer/retail_data.html",{"file_name":file_name_from_database, "selection_value":0, "df_lapsed":df_lapsed, "df_average":df_average,"df_shark":df_shark,"df_whales":df_whales, "title":"Segmented Customer Based"})



def plot_to_base64(plt):
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    string = base64.b64encode(buf.read())
    uri = urllib.parse.quote(string)
    return uri


def analytics_dashboard(request):
    RFM = pd.DataFrame(list(Customer.objects.all().values()))
    
    if len(Customer.objects.all().values()) >= 50:
        
        # Pie chart for Customer Segmentation Distribution
        plt.figure(figsize=(6, 4))
        RFM['group'].value_counts().plot.pie(autopct='%1.1f%%', startangle=140)
        plt.title('Customer Segmentation Distribution')
        pie_chart = plot_to_base64(plt)
        plt.clf()

        # Box plots for RFM Analysis
        plt.figure(figsize=(6, 4))
        sns.boxplot(x='group', y='recency', data=RFM)
        plt.title('Recency Distribution by Customer Group')
        recency_plot = plot_to_base64(plt)
        plt.clf()

        plt.figure(figsize=(6, 4))
        sns.boxplot(x='group', y='frequency', data=RFM)
        plt.title('Frequency Distribution by Customer Group')
        frequency_plot = plot_to_base64(plt)
        plt.clf()

        plt.figure(figsize=(6, 4))
        sns.boxplot(x='group', y='monetary', data=RFM)
        plt.title('Monetary Distribution by Customer Group')
        monetary_plot = plot_to_base64(plt)
        plt.clf()

        plt.figure(figsize=(6, 4))
        fig = px.scatter(RFM, x='customer_id', y='group', color='group', title="Customer Groups",
                     labels={'customer_id': 'Customer ID', 'group': 'Customer Group'},
                     hover_data=['recency', 'frequency', 'monetary'])

        fig.update_layout(width=600, height=350)

        graph_html = pio.to_html(fig, full_html=False)

        context = {
            'pie_chart': pie_chart,
            'recency_plot': recency_plot,
            'frequency_plot': frequency_plot,
            'monetary_plot': monetary_plot,
            'graph_html': graph_html,
        }

        return render(request, 'customer/analytics_dashboard.html', context)

    else:
        return render(request, "customer/dashboard_analytics.html",{"message":"Instances of data is too less for analytics"})


def error_page(request, message):
    return render(request, "customer/error.html", {"message": message})


