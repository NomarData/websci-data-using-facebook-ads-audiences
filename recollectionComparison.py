import pandas as pd
import ast
import sys
from scipy.stats import pearsonr as pearsonScipy
from scipy.stats import spearmanr as spearmanScipy
from scipy.stats import bayes_mvs
from numpy import inf

def pearsonr(v1,v2):
    rho,pvalue = pearsonScipy(v1,v2)
    if pd.isnull(rho):
        rho = 0
        pvalue = 1
    return (rho,pvalue)

def spearmanr(v1,v2):
    rho,pvalue = spearmanScipy(v1,v2)
    if pd.isnull(rho):
        rho = 0
        pvalue = 1
    return (rho,pvalue)

countries = {}
COUNTRY_CODE = "Country Code"
INTEREST = "Interest"
PREVIOUS_AUDIENCE = "Previous Audience"
CURRENT_AUDIENCE = "Current Audience"
ABSOLUTE_VARIATION = "Absolute Variation"
RELATIVE_VARIATION = "Relative Variation"
NORMALIZED_VARIATION = "Normalized Variation"
PREVIOUS_RELATIVE_AUDIENCE = "Previous Relative Audience"
CURRENT_RELATIVE_AUDIENCE = "Current Relative Audience"
PREVIOUS_RANK = "previousRank"
CURRENT_RANK = "currentRank"
SPEARMAN_CORRELATION = "Spearman Correlation"
SPEARMAN_P_VALUE = "S. p-value"
PEARSON_CORRELATION = "Pearson Correlation"
PEARSON_P_VALUE = "P. p-value"
COUNTRIES_CONSIDERED = "Countries"
VALUE = "value"
GENDER = "gender"
AGE = "age"
MIN_AGE = "min_age"
MAX_AGE = "max_age"
PREVIOUS_RANK_RELATIVE = "Previous Rank Relative Audience"
CURRENT_RANK_RELATIVE = "Current Rank Relative Audience"
PREVIOUS_AVERAGE_AUDIENCE = "Previous Average Audience"
CURRENT_AVERAGE_AUDIENCE = "Current Average Audience"
SIGNED_VARIATION = "Signed Variation"
SIGNED_VARIATION_PERCENTAGE = "Signed Variation Percentage"

AUDIENCE_DATA = {VALUE: 0, GENDER: 0, MIN_AGE: 0, MAX_AGE: 0}

class CountryData():
    def __init__(self, code):
        self.code = code
        self.audiences = {}
        self.audiences[PREVIOUS_AUDIENCE] = []
        self.audiences[CURRENT_AUDIENCE] = []
        self.audiences[RELATIVE_VARIATION] = []
        self.audiences[ABSOLUTE_VARIATION] = []
        self.audiences[SIGNED_VARIATION] = []
        self.audiences[SIGNED_VARIATION_PERCENTAGE] = []
        self.topics_audiences = []

    def set_old_audience(self, audience):
        self.audiences[PREVIOUS_AUDIENCE] = audience

    def set_new_audience(self, audience):
        self.audiences[CURRENT_AUDIENCE] = audience

    def add_new_topic(self, new_topic):
        self.topics_audiences.append(new_topic)

    def get_topic_by_ids(self, ids):
        for topic in self.topics_audiences:
            topic_ids = topic["ids"]
            if set(topic_ids) == set(topic_ids) & set(ids) and set(ids) == set(topic_ids) & set(ids):
                return topic
        return None

    def get_data_frame_rows(self):
        row_template = {COUNTRY_CODE: None, INTEREST: None, GENDER:None, MIN_AGE:None, MAX_AGE:None, PREVIOUS_AUDIENCE : None, CURRENT_AUDIENCE : None, ABSOLUTE_VARIATION : None, RELATIVE_VARIATION : None, NORMALIZED_VARIATION: None}
        rows = []
        for new_audience_data in self.audiences[CURRENT_AUDIENCE]:
            old_audience_data = self.get_audience_data_given_audience_data_and_type(new_audience_data, PREVIOUS_AUDIENCE)
            country_audience_row = row_template.copy()
            country_audience_row[COUNTRY_CODE] = self.code
            country_audience_row[GENDER] = new_audience_data[GENDER]
            country_audience_row[MIN_AGE] = new_audience_data[MIN_AGE]
            country_audience_row[MAX_AGE] = new_audience_data[MAX_AGE]
            country_audience_row[PREVIOUS_AUDIENCE] = old_audience_data[VALUE]
            country_audience_row[CURRENT_AUDIENCE] = new_audience_data[VALUE]
            country_audience_row[ABSOLUTE_VARIATION] = get_audience_data_from_audience_data_list_given_audience_data_as_query(new_audience_data, self.audiences[ABSOLUTE_VARIATION])[VALUE]
            country_audience_row[RELATIVE_VARIATION] = get_audience_data_from_audience_data_list_given_audience_data_as_query(new_audience_data, self.audiences[RELATIVE_VARIATION])[VALUE]
            rows.append(country_audience_row)

        for topic in self.topics_audiences:
            for new_audience_data in topic[CURRENT_AUDIENCE]:
                old_audience_data = self.get_audience_data_given_audience_data_audience_type_topic(new_audience_data,topic, PREVIOUS_AUDIENCE)
                country_audience_row = row_template.copy()
                country_audience_row[COUNTRY_CODE] = self.code
                country_audience_row[INTEREST] = topic[INTEREST]
                country_audience_row[GENDER] = new_audience_data[GENDER]
                country_audience_row[MIN_AGE] = new_audience_data[MIN_AGE]
                country_audience_row[MAX_AGE] = new_audience_data[MAX_AGE]
                country_audience_row[PREVIOUS_AUDIENCE] = old_audience_data[VALUE]
                country_audience_row[CURRENT_AUDIENCE] = new_audience_data[VALUE]
                country_audience_row[ABSOLUTE_VARIATION] = get_audience_data_from_audience_data_list_given_audience_data_as_query(new_audience_data, topic[ABSOLUTE_VARIATION])[VALUE]
                country_audience_row[RELATIVE_VARIATION] = get_audience_data_from_audience_data_list_given_audience_data_as_query(new_audience_data, topic[RELATIVE_VARIATION])[VALUE]
                country_audience_row[NORMALIZED_VARIATION] = get_audience_data_from_audience_data_list_given_audience_data_as_query(new_audience_data, topic[NORMALIZED_VARIATION])[VALUE]
                rows.append(country_audience_row)
        return rows

    def set_new_audience_topic_and_name(self, new_audience, ids, name):
        topic = self.get_topic_by_ids(ids)
        topic[CURRENT_AUDIENCE] = new_audience
        topic[CURRENT_RELATIVE_AUDIENCE] = new_audience / float(self.audiences[CURRENT_AUDIENCE])
        topic[INTEREST] = name

    def get_audience_data_given_audience_data_and_type(self, audience_data, audience_type):
        return get_audience_data_from_audience_data_list_given_audience_data_as_query(audience_data, self.audiences[audience_type])

    def get_audience_data_given_audience_data_audience_type_topic(self, audience_data, topic, audience_type):
        return get_audience_data_from_audience_data_list_given_audience_data_as_query(audience_data,topic[audience_type])

    def get_normalized_audience_given_topic_and_audience_data_query(self, audience_data_query, topic):
        new_topic_audience_data = get_audience_data_from_audience_data_list_given_audience_data_as_query(audience_data_query, topic[CURRENT_AUDIENCE])
        old_topic_audience_data = get_audience_data_from_audience_data_list_given_audience_data_as_query(audience_data_query, topic[PREVIOUS_AUDIENCE])
        new_audience_data = get_audience_data_from_audience_data_list_given_audience_data_as_query(audience_data_query, self.audiences[CURRENT_AUDIENCE])
        old_audience_data = get_audience_data_from_audience_data_list_given_audience_data_as_query(audience_data_query, self.audiences[PREVIOUS_AUDIENCE])
        old_topic_audience = old_topic_audience_data[VALUE]
        new_topic_audience = new_topic_audience_data[VALUE]
        old_audience = old_audience_data[VALUE]
        new_audience = new_audience_data[VALUE]
        return ((old_topic_audience / old_audience) - (new_topic_audience / new_audience) / (old_topic_audience / old_audience)) if (old_audience != 0 and new_audience != 0 and old_topic_audience != 0 ) else None

    def calculate_relative_audiences(self):
        # Calculating variation in audiences per interests
        for topic in self.topics_audiences:
            topic[PREVIOUS_RELATIVE_AUDIENCE] = []
            topic[CURRENT_RELATIVE_AUDIENCE] = []

            old_topic_audiences_data = topic[PREVIOUS_AUDIENCE]
            for old_topic_audience_data in old_topic_audiences_data:
                relative_old_audience_data = old_topic_audience_data.copy()
                old_audience_data = self.get_audience_data_given_audience_data_and_type(old_topic_audience_data, PREVIOUS_AUDIENCE)
                relative_old_audience_data[VALUE] = old_topic_audience_data[VALUE] / old_audience_data[VALUE]
                topic[PREVIOUS_RELATIVE_AUDIENCE].append(relative_old_audience_data)

            new_topic_audiences_data = topic[CURRENT_AUDIENCE]
            for new_topic_audience_data in new_topic_audiences_data:
                relative_new_audience_data = new_topic_audience_data.copy()
                new_audience_data = self.get_audience_data_given_audience_data_and_type(new_topic_audience_data,
                                                                                        CURRENT_AUDIENCE)
                relative_new_audience_data[VALUE] = new_topic_audience_data[VALUE] / new_audience_data[VALUE]
                topic[CURRENT_RELATIVE_AUDIENCE].append(relative_new_audience_data)

    def process_comparison(self):
        #Calculating variation in audiences without interests
        new_country_audiences = self.audiences[CURRENT_AUDIENCE]

        for new_audience_data in new_country_audiences:
            past_audience_data = self.get_audience_data_given_audience_data_and_type(new_audience_data, PREVIOUS_AUDIENCE)
            #Get relative variation
            relative_audience_data = new_audience_data.copy()
            relative_audience_data[VALUE] = abs(past_audience_data[VALUE] - new_audience_data[VALUE]) / past_audience_data[VALUE] if past_audience_data[VALUE] != 0 else None
            self.audiences[RELATIVE_VARIATION].append(relative_audience_data)
            # Get absolute variation
            absolute_variation_data = new_audience_data.copy()
            absolute_variation_data[VALUE] = abs(past_audience_data[VALUE] - new_audience_data[VALUE])
            self.audiences[ABSOLUTE_VARIATION].append(absolute_variation_data)

            # Signed Audience Variation
            signed_variation_audience_data = new_audience_data.copy()
            signed_variation_audience_data[VALUE] =  new_audience_data[VALUE] - past_audience_data[VALUE]
            self.audiences[SIGNED_VARIATION].append(signed_variation_audience_data)

            # Signed Audience Variation Percentage
            signed_variation_percentage_topic_audience_data = new_audience_data.copy()
            signed_variation_percentage_topic_audience_data[VALUE] = 1 - (new_audience_data[VALUE] / past_audience_data[VALUE])
            self.audiences[SIGNED_VARIATION_PERCENTAGE].append(signed_variation_percentage_topic_audience_data)

        # Calculating variation in audiences per interests
        for topic in self.topics_audiences:
            topic_new_audiences = topic[CURRENT_AUDIENCE]
            for new_topic_audience in topic_new_audiences:
                old_audience_data = self.get_audience_data_given_audience_data_and_type(new_topic_audience, PREVIOUS_AUDIENCE)
                new_audience_data = self.get_audience_data_given_audience_data_and_type(new_topic_audience, CURRENT_AUDIENCE)

                past_topic_audience = self.get_audience_data_given_audience_data_audience_type_topic(new_topic_audience, topic, PREVIOUS_AUDIENCE)
                print "Calculating", new_topic_audience, past_topic_audience
                # Get relative variation
                relative_topic_audience_data = new_topic_audience.copy()
                relative_topic_audience_data[VALUE] = abs(past_topic_audience[VALUE] - new_topic_audience[VALUE]) / past_topic_audience[VALUE] if past_topic_audience[VALUE] != 0 else None
                topic[RELATIVE_VARIATION].append(relative_topic_audience_data)
                # Get absolute variation
                absolute_topic_audience_data = new_topic_audience.copy()
                absolute_topic_audience_data[VALUE] = abs(past_topic_audience[VALUE] - new_topic_audience[VALUE])
                topic[ABSOLUTE_VARIATION].append(absolute_topic_audience_data)

                #Signed Audience Variation
                signed_variation_topic_audience_data = new_topic_audience.copy()
                signed_variation_topic_audience_data[VALUE] =  new_topic_audience[VALUE] - past_topic_audience[VALUE]
                topic[SIGNED_VARIATION].append(signed_variation_topic_audience_data)

                # Signed Audience Variation Percentage
                signed_variation_percentage_topic_audience_data = new_topic_audience.copy()
                # signed_variation_percentage_topic_audience_data[VALUE] = (new_topic_audience[VALUE]/new_audience_data[VALUE] - past_topic_audience[VALUE]/old_audience_data[VALUE]) / (past_topic_audience[VALUE]/old_audience_data[VALUE])
                signed_variation_percentage_topic_audience_data[VALUE] = self.get_normalized_audience_given_topic_and_audience_data_query(new_topic_audience, topic)
                topic[SIGNED_VARIATION_PERCENTAGE].append(signed_variation_percentage_topic_audience_data)

                # Get normalized variation
                normalized_topic_audience_data = new_topic_audience.copy()
                normalized_topic_audience_data[VALUE] = abs(self.get_normalized_audience_given_topic_and_audience_data_query(new_topic_audience, topic)) if self.get_normalized_audience_given_topic_and_audience_data_query(new_topic_audience, topic) else None
                topic[NORMALIZED_VARIATION].append(normalized_topic_audience_data)

    def get_country_code_and_relative_audience(self,topic_ids, AUDIENCE_TYPE):
        topic = self.get_topic_by_ids(topic_ids)
        relative_audience = topic[AUDIENCE_TYPE]
        return (self.code, relative_audience)

    def get_audience_given_age_gender(self, min, max, gender, audience_type):
        for audience in self.audiences[audience_type]:
            if audience[MIN_AGE] == min and audience[MAX_AGE] == max and audience[GENDER] == gender:
                return audience[VALUE]
        raise Exception("There is no audience for: " + str(self.code) + " " + str(min) + " " + str(max) + " " + str(gender) + " " + str(audience_type) )

    def add_country_audience(self, audience_data, audience_type):
        self.audiences[audience_type].append(audience_data)



    def __str__(self):
        country_str = self.code + ":" + str(self.audiences) + "\n"
        for topic in self.topics_audiences:
            country_str += str(topic[INTEREST]) + ":" +  str(topic["ids"]) + "\t From: " + str(topic[PREVIOUS_AUDIENCE]) + " To: " + str(topic[CURRENT_AUDIENCE]) + "\n"
        return country_str

    def add_audience_data_given_topic_and_country(self, topic, audience_data, audience_type):
        topic[audience_type].append(audience_data)
        # relative_audience_data = audience_data.copy()
        # relative_audience_data[VALUE] = audience_data[VALUE] / float(self.get_audience_given_age_gender(audience_data[MIN_AGE], audience_data[MAX_AGE], audience_data[GENDER], audience_type))
        # topic[PREVIOUS_RELATIVE_AUDIENCE].append(relative_audience_data)

    @staticmethod
    def get_country(code):
        for country in countries.values():
            if country.code == code:
                return country
        raise Exception("No country with code: " + code)

    @staticmethod
    def get_all_countries_code():
        countries_code = []
        for country in countries.values():
            countries_code.append(country.code)
        return countries_code


def get_audience_data_from_audience_data_list_given_audience_data_as_query(audience_data_query, audience_data_list):
    keys = AUDIENCE_DATA.keys()
    keys.remove(VALUE)
    for aux_audience_data in audience_data_list:
        for key in keys:
            if not pd.isnull(audience_data_query[key]) and aux_audience_data[key] != audience_data_query[key]:
                break
            if pd.isnull(audience_data_query[key]) and not pd.isnull(aux_audience_data[key]):
                break
            if key == keys[-1]:
                print "Match: ", audience_data_query, aux_audience_data
                return aux_audience_data
    raise Exception("Should have a match for:" + str(audience_data_query))

def get_or_create_country_if_need(row):
    country_code = row["country_code"]
    if not country_code in countries.keys():
        country = CountryData(country_code)
        countries[country.code] = country
        return country
    else:
        return countries[country_code]

def create_topic_if_need(row, country):
    # Check if have, if not create another one
    ids = ast.literal_eval(row["interest_query"])["or"]
    topic = country.get_topic_by_ids(ids)
    if not topic:
        new_topic = {
            "ids": ids,
            INTEREST: "unamed",
            PREVIOUS_AUDIENCE: [],
            PREVIOUS_RELATIVE_AUDIENCE: [],
            CURRENT_AUDIENCE: [],
            CURRENT_RELATIVE_AUDIENCE: [],
            RELATIVE_VARIATION: [],
            NORMALIZED_VARIATION: [],
            ABSOLUTE_VARIATION: [],
            SIGNED_VARIATION: [],
            SIGNED_VARIATION_PERCENTAGE : []
        }
        country.add_new_topic(new_topic)
        topic = new_topic
    return topic

def add_old_row(row):
    print "Old Data: ", row.name
    audience_data = AUDIENCE_DATA.copy()
    audience_data[MIN_AGE] = row[MIN_AGE]
    audience_data[MAX_AGE] = row[MAX_AGE]
    audience_data[VALUE] = float(row["audience"]) if row["audience"] > 20 else 0
    audience_data[GENDER] = row["gender"]

    if pd.isnull(row["interest_query"]):
        country = get_or_create_country_if_need(row)
        country.add_country_audience(audience_data, PREVIOUS_AUDIENCE)
    else:
        country = get_or_create_country_if_need(row)
        topic = create_topic_if_need(row, country)
        country.add_audience_data_given_topic_and_country(topic, audience_data, PREVIOUS_AUDIENCE)

def add_new_row(row):
    print "New Data: ", row.name
    age_range = ast.literal_eval(row["ages_ranges"])
    if age_range["min"] == 20:
        return
    country_code = ast.literal_eval(row["geo_locations"])["values"][0]
    audience_data = AUDIENCE_DATA.copy()
    audience_data[MIN_AGE] = age_range["min"]
    audience_data[MAX_AGE] = age_range["max"] if "max" in age_range.keys() else None
    audience_data[VALUE] = float(row["audience"]) if row["audience"] > 20 else 0
    audience_data[GENDER] = row["genders"]

    if pd.isnull(row["interests"]):
        country = CountryData.get_country(country_code)
        country.add_country_audience(audience_data, CURRENT_AUDIENCE)
    else:
        ids = ast.literal_eval(row["interests"])["or"]
        name = ast.literal_eval(row["interests"])["name"]
        country = CountryData.get_country(country_code)
        topic = country.get_topic_by_ids(ids)
        topic[INTEREST] = name
        country.add_audience_data_given_topic_and_country(topic, audience_data, CURRENT_AUDIENCE)


def get_country_rank_given_topic_ids(topic_ids, AUDIENCE_TYPE):
    country_codes_relative_audience = map(lambda country: country.get_country_code_and_relative_audience(topic_ids, AUDIENCE_TYPE) , countries.values())
    country_codes_relative_audience = sorted(country_codes_relative_audience, key=lambda x: x[1])
    return country_codes_relative_audience


def calculate_spearman_correlation(original_topic_data, coutries_codes, min_initial_relative_audience = 0):
    topic = original_topic_data.copy()

    #Filter minimum relative audience
    countries_to_remove = map(lambda rank_item: rank_item[0], filter(lambda rank_item: rank_item[1] < min_initial_relative_audience, topic[PREVIOUS_RANK]))

    previous_rank = filter(lambda rank_item: rank_item[0] in coutries_codes and not rank_item[0] in countries_to_remove, topic[PREVIOUS_RANK])
    current_rank = filter(lambda rank_item: rank_item[0] in coutries_codes and not rank_item[0] in countries_to_remove, topic[CURRENT_RANK])

    previous_rank_labels = map(lambda rank_item: rank_item[0], previous_rank)
    current_rank_labels = map(lambda rank_item: rank_item[0], current_rank)

    previous_rank_positions = range(0, len(previous_rank_labels))
    current_rank_positions = []
    for country_code in previous_rank_labels:
        current_position = current_rank_labels.index(country_code)
        current_rank_positions.append(current_position)
    rho, pvalue = spearmanr(previous_rank_positions, current_rank_positions)
    return (rho, pvalue, set(coutries_codes) - set(countries_to_remove))

def calculate_spearman_correlation_age_rank(topic,previous_rank, current_rank):
    rho,pvalue = spearmanr(topic[previous_rank], topic[current_rank])
    return (rho,pvalue)

def calculate_pearson_correlation_age_rank(topic,previous_rank, current_rank):
    rho, pvalue = pearsonr(topic[previous_rank], topic[current_rank])
    return rho, pvalue


# def build_correlation_dataframe_given_countries_topics(topics, countries, name, writer,min_initial_relative_audience):
#     if min_initial_relative_audience > 0:
#         name = str(min_initial_relative_audience) + "_" + name
#
#     df = pd.DataFrame(columns=[INTEREST, SPEARMAN_CORRELATION, SPEARMAN_P_VALUE, COUNTRIES_CONSIDERED])
#     rows = []
#     for topic in topics:
#         spearman_correlation, p_value, countries_considered = calculate_spearman_correlation(topic, countries, min_initial_relative_audience)
#         df_row = {INTEREST: topic[INTEREST], SPEARMAN_CORRELATION: spearman_correlation, SPEARMAN_P_VALUE : p_value, COUNTRIES_CONSIDERED: "-".join(countries_considered)}
#         rows.append(df_row)
#     df = df.append(rows)
#     df = df.sort_values(by=SPEARMAN_CORRELATION, ascending=False)
#
#     workbook = writer.book
#     format1 = workbook.add_format({'num_format': '#,##0.0000000000'})
#     df.to_excel(writer, sheet_name=name)
#     writer.sheets[name].set_column('A:D', 18, format1)
#     return df

def get_current_interest_audience(interest):
    if not pd.isnull(interest):
        audiences = new_data_18p[new_data_18p[INTEREST] == interest]["audience"].values.tolist()
        mean = reduce(lambda x, y: x + y, audiences) / float(len(audiences))
        return mean
    else:
        return None



def print_summary_results(df, writer, sheet_name):
    workbook = writer.book
    worksheet = workbook.add_worksheet(sheet_name)
    bold = workbook.add_format({'bold': 1})


    # Number of Topics below 0.7 of correlation
    less_70 = len(df[df[SPEARMAN_CORRELATION] < 0.7])
    df = df[~pd.isnull(df[SPEARMAN_CORRELATION])]
    overall = "Number of spearman correlations below 0.7: " + str(less_70) + " which is " + str(less_70/float(len(df))*100) + "%"
    worksheet.write('A1', overall, bold)
    interests = df[INTEREST].unique().tolist()
    worksheet.write('A2', "Summary by interest:", bold)
    worksheet.write('A3', "Name", bold)
    worksheet.write('B3', "Mean", bold)
    worksheet.write('C3', "I. Conf", bold)
    worksheet.write('D3', "Std", bold)
    worksheet.write('E3', "Average Audience 18+", bold)
    row = 3
    col = 0
    means_correlations = {"means":[], "audiences":[]}
    for interest in interests:
        interest_audience = get_current_interest_audience(interest)
        mean = -10000
        if len(df[df[INTEREST] == interest][SPEARMAN_CORRELATION].unique()) == 1:
            mean = df[df[INTEREST] == interest][SPEARMAN_CORRELATION].unique()[0]
            worksheet.write_string(row, col, str(interest))
            worksheet.write_number(row, col + 1, mean)
            worksheet.write_string(row, col + 2, "-")
            worksheet.write_string(row, col + 3, "-")
            worksheet.write_number(row, col + 4, interest_audience if interest_audience else 0)
        else:
            if pd.isnull(interest):
                meanStat, var, std = bayes_mvs(df[pd.isnull(df[INTEREST])][SPEARMAN_CORRELATION].values.tolist())
            else:
                meanStat, var, std = bayes_mvs(df[df[INTEREST] == interest][SPEARMAN_CORRELATION].values.tolist())
            mean = meanStat[0]
            worksheet.write_string(row, col, str(interest))
            worksheet.write_number(row, col + 1, mean)
            worksheet.write_string(row, col + 2, str(meanStat[1]))
            worksheet.write(row, col + 3, convertNanInfToString(std[0]))
            worksheet.write_number(row, col + 4, interest_audience if interest_audience else 0)
        if interest:
            means_correlations["means"].append(mean)
            means_correlations["audiences"].append(interest_audience)
        row += 1
    row+=1
    # Write Correlation
    worksheet.write_string(row, col, "Correlation Mean vs Avg. Audience", bold)
    worksheet.write_string(row, col + 1, "Correlation", bold)
    worksheet.write_string(row, col + 2, "P-Value", bold)
    row += 1
    worksheet.write_string(row, col, "Spearman", bold)
    worksheet.write_number(row, col + 1, spearmanr(means_correlations["means"], means_correlations["audiences"])[0])
    worksheet.write_number(row, col + 2, spearmanr(means_correlations["means"], means_correlations["audiences"])[1])
    row += 1
    worksheet.write_string(row, col, "Pearson", bold)
    worksheet.write_number(row, col + 1, pearsonr(means_correlations["means"], means_correlations["audiences"])[0])
    worksheet.write_number(row, col + 2, pearsonr(means_correlations["means"], means_correlations["audiences"])[1])
    row += 2

    # COuntry Correrlations
    worksheet.write_string(row, col , "Summary by country:",bold)
    row +=1
    for countryCode in countries:
        mean, var, std = bayes_mvs(df[df[COUNTRY_CODE] == countryCode][SPEARMAN_CORRELATION].values.tolist())
        worksheet.write_string(row, col, countryCode)
        worksheet.write_number(row, col + 1, mean[0])
        worksheet.write_string(row, col + 2, str(mean[1]))
        worksheet.write_number(row, col + 3, std[0])
        row += 1

# def build_correlation_dataframe_given_topics_rank_by_gender(topics, name, writer, min_initial_relative_audience):
#     if min_initial_relative_audience > 0:
#         name = str(min_initial_relative_audience) + "_" + name
#
#     df = pd.DataFrame(columns=[INTEREST, COUNTRY_CODE, GENDER, SPEARMAN_CORRELATION, SPEARMAN_P_VALUE, PREVIOUS_AUDIENCE, CURRENT_AUDIENCE])
#     rows = []
#     for topic in topics:
#         spearman_correlation, spearman_p_value = calculate_spearman_correlation_age_rank(topic, PREVIOUS_RANK_RELATIVE, CURRENT_RANK_RELATIVE)
#         pearson_correlation, pearson_p_value = calculate_pearson_correlation_age_rank(topic, PREVIOUS_RANK_RELATIVE, CURRENT_RANK_RELATIVE)
#         previous_average = reduce(lambda x, y: x + y, topic[PREVIOUS_RANK]) / len(PREVIOUS_RANK)
#         current_average = reduce(lambda x, y: x + y, topic[CURRENT_RANK]) / len(CURRENT_RANK)
#         df_row = {INTEREST: topic[INTEREST], COUNTRY_CODE: topic[COUNTRY_CODE], SPEARMAN_CORRELATION: spearman_correlation, SPEARMAN_P_VALUE : spearman_p_value, PEARSON_CORRELATION: pearson_correlation, PEARSON_P_VALUE : pearson_p_value, PREVIOUS_RANK_RELATIVE: topic[PREVIOUS_RANK_RELATIVE], CURRENT_RANK_RELATIVE: topic[CURRENT_RANK_RELATIVE], PREVIOUS_AVERAGE_AUDIENCE: previous_average, CURRENT_AVERAGE_AUDIENCE: current_average}
#         rows.append(df_row)
#         topic[SPEARMAN_CORRELATION] = spearman_correlation
#         topic[SPEARMAN_P_VALUE] = spearman_p_value
#         topic[PEARSON_CORRELATION] = pearson_correlation
#         topic[PEARSON_P_VALUE] = pearson_p_value
#         topic[PREVIOUS_AVERAGE_AUDIENCE] = previous_average
#         topic[CURRENT_AVERAGE_AUDIENCE] = current_average
#
#     df = df.append(rows)
#     df = df.sort_values(by=[SPEARMAN_CORRELATION, INTEREST], ascending=False)
#
#     workbook = writer.book
#     format1 = workbook.add_format({'num_format': '#,##0.0000000000'})
#     df[[INTEREST, COUNTRY_CODE, GENDER, SPEARMAN_CORRELATION, SPEARMAN_P_VALUE, PEARSON_CORRELATION, PEARSON_P_VALUE, PREVIOUS_RANK_RELATIVE, CURRENT_RANK_RELATIVE, PREVIOUS_AVERAGE_AUDIENCE, CURRENT_AVERAGE_AUDIENCE]].to_excel(writer, sheet_name=name)
#     print_summary_results(df,writer,"Summary")
#     writer.sheets[name].set_column('A:F', 18, format1)
#     writer.sheets[name].set_column('G:J', 30, format1)
#     return df
def convertNanInfToString(value):
    if pd.isnull(value):
        return "None"
    if value == inf:
        return "Inf."
    return value

def compare_two_contries_movement(country1, country2, writer):
    countries_side_by_side = []
    topics_correlations = []
    workbook = writer.book
    worksheet = workbook.add_worksheet(country1.code + " vs. " + country2.code)
    bold = workbook.add_format({'bold': 1})
    row = 0
    col = 0
    worksheet.write(row, col, "Interest", bold)
    worksheet.write(row, col+1, "Gender", bold)
    worksheet.write(row, col+2, "Min Age", bold)
    worksheet.write(row, col+3, "Max Age", bold)
    worksheet.write(row, col + 4, country1.code + " Relative Variation", bold)
    worksheet.write(row, col + 5, country2.code + " Relative Variation", bold)
    worksheet.write(row, col + 6, country1.code + " Previous Audience", bold)
    worksheet.write(row, col + 7, country1.code + " Current Audience", bold)
    worksheet.write(row, col + 8, country2.code + " Previous Audience", bold)
    worksheet.write(row, col + 9, country2.code + " Current Audience", bold)
    row += 1
    for topic in country1.topics_audiences:
        countries_side_by_side_by_topic = []
        for audience_data in topic[CURRENT_AUDIENCE]:
            interest = topic[INTEREST]
            gender = audience_data[GENDER]
            min_age = audience_data[MIN_AGE]
            max_age = audience_data[MAX_AGE]
            ids = topic["ids"]
            c1_variation = get_audience_data_from_audience_data_list_given_audience_data_as_query(audience_data, country1.get_topic_by_ids(ids)[SIGNED_VARIATION_PERCENTAGE])[VALUE]
            c1_old_value = get_audience_data_from_audience_data_list_given_audience_data_as_query(audience_data, country1.get_topic_by_ids(ids)[PREVIOUS_AUDIENCE])[VALUE]
            c1_new_value = get_audience_data_from_audience_data_list_given_audience_data_as_query(audience_data, country1.get_topic_by_ids( ids)[CURRENT_AUDIENCE])[VALUE]

            c2_variation = get_audience_data_from_audience_data_list_given_audience_data_as_query(audience_data, country2.get_topic_by_ids(ids)[SIGNED_VARIATION_PERCENTAGE])[VALUE]
            c2_old_value = get_audience_data_from_audience_data_list_given_audience_data_as_query(audience_data,country2.get_topic_by_ids(ids)[PREVIOUS_AUDIENCE])[VALUE]
            c2_new_value = get_audience_data_from_audience_data_list_given_audience_data_as_query(audience_data,country2.get_topic_by_ids(ids)[CURRENT_AUDIENCE])[VALUE]
            if pd.isnull(c1_variation) or pd.isnull(c2_variation):
                continue
            tuple_row = (interest,gender,min_age,max_age,c1_variation,c2_variation,c1_old_value,c1_new_value,c2_old_value,c2_new_value)
            countries_side_by_side_by_topic.append(tuple_row)
            worksheet.write(row, col, tuple_row[0])
            worksheet.write(row, col + 1, tuple_row[1])
            worksheet.write(row, col + 2, tuple_row[2])
            worksheet.write(row, col + 3, tuple_row[3])
            worksheet.write(row, col + 4, tuple_row[4])
            worksheet.write(row, col + 5, tuple_row[5])
            worksheet.write(row, col + 6, tuple_row[6])
            worksheet.write(row, col + 7, tuple_row[7])
            worksheet.write(row, col + 8, tuple_row[8])
            worksheet.write(row, col + 9, tuple_row[9])
            row += 1

        worksheet.write(row, col, "Correlation in Topic", bold)
        worksheet.write(row, col + 1, "Value", bold)
        worksheet.write(row, col + 2, "P-value", bold)
        row += 1
        rhop, pvaluep = pearsonr(map(lambda row: row[4], countries_side_by_side_by_topic),
                                 map(lambda row: row[5], countries_side_by_side_by_topic))
        worksheet.write(row, col, "Pearson")
        worksheet.write(row, col + 1, convertNanInfToString(rhop))
        worksheet.write(row, col + 2, convertNanInfToString(pvaluep))
        row += 1
        rhos, pvalues = spearmanr(map(lambda row: row[4], countries_side_by_side_by_topic),
                                  map(lambda row: row[5], countries_side_by_side_by_topic))
        worksheet.write(row, col, "Spearman")
        worksheet.write(row, col + 1, convertNanInfToString(rhos))
        worksheet.write(row, col + 2, convertNanInfToString(pvalues))
        countries_side_by_side += countries_side_by_side_by_topic
        topics_correlations.append((interest, rhop, pvaluep, rhos, pvalues))
        row += 2
    row += 1
    worksheet.write(row, col, "Summary", bold)
    row += 1
    worksheet.write(row, col, "Correlation", bold)
    worksheet.write(row, col + 1, "Value", bold)
    worksheet.write(row, col + 2, "P-value", bold)
    row += 1
    rhop,pvaluep = pearsonr(map(lambda row: row[4], countries_side_by_side), map(lambda row: row[5], countries_side_by_side))
    worksheet.write(row, col, "Pearson")
    worksheet.write(row, col + 1, rhop)
    worksheet.write(row, col + 2, pvaluep)
    row += 1
    rhos, pvalues = spearmanr(map(lambda row: row[4], countries_side_by_side), map(lambda row: row[5], countries_side_by_side))
    worksheet.write(row, col, "Spearman")
    worksheet.write(row, col + 1, rhos)
    worksheet.write(row, col + 2, pvalues)
    row += 1
    worksheet.write(row, col, "Interest", bold)
    worksheet.write(row, col + 1, "Pearson", bold)
    worksheet.write(row, col + 2, "P-Value", bold)
    worksheet.write(row, col + 3, "Spearman", bold)
    worksheet.write(row, col + 4, "P-value", bold)
    row += 1
    for topics_correlation in topics_correlations:
        worksheet.write(row, col, topics_correlation[0])
        worksheet.write(row, col + 1, convertNanInfToString(topics_correlation[1]))
        worksheet.write(row, col + 2, convertNanInfToString(topics_correlation[2]))
        worksheet.write(row, col + 3, convertNanInfToString(topics_correlation[3]))
        worksheet.write(row, col + 4, convertNanInfToString(topics_correlation[4]))
        row += 1




def build_correlation_dataframe_given_topics(topics, name, writer, min_initial_relative_audience):
    if min_initial_relative_audience > 0:
        name = str(min_initial_relative_audience) + "_" + name

    df = pd.DataFrame(columns=[INTEREST, COUNTRY_CODE, GENDER, SPEARMAN_CORRELATION, SPEARMAN_P_VALUE, PREVIOUS_AUDIENCE, CURRENT_AUDIENCE])
    rows = []
    for topic in topics:
        spearman_correlation, spearman_p_value = calculate_spearman_correlation_age_rank(topic, PREVIOUS_RANK_RELATIVE, CURRENT_RANK_RELATIVE)
        pearson_correlation, pearson_p_value = calculate_pearson_correlation_age_rank(topic, PREVIOUS_RANK_RELATIVE, CURRENT_RANK_RELATIVE)
        previous_average = reduce(lambda x, y: x + y, topic[PREVIOUS_RANK]) / len(PREVIOUS_RANK)
        current_average = reduce(lambda x, y: x + y, topic[CURRENT_RANK]) / len(CURRENT_RANK)
        df_row = {INTEREST: topic[INTEREST], COUNTRY_CODE: topic[COUNTRY_CODE], GENDER: topic[GENDER] if GENDER in topic else None, SPEARMAN_CORRELATION: spearman_correlation, SPEARMAN_P_VALUE : spearman_p_value, PEARSON_CORRELATION: pearson_correlation, PEARSON_P_VALUE : pearson_p_value, PREVIOUS_RANK_RELATIVE: topic[PREVIOUS_RANK_RELATIVE], CURRENT_RANK_RELATIVE: topic[CURRENT_RANK_RELATIVE], PREVIOUS_AVERAGE_AUDIENCE: previous_average, CURRENT_AVERAGE_AUDIENCE: current_average, PREVIOUS_RANK: topic[PREVIOUS_RANK], CURRENT_RANK: topic[CURRENT_RANK]}
        rows.append(df_row)
        topic[SPEARMAN_CORRELATION] = spearman_correlation
        topic[SPEARMAN_P_VALUE] = spearman_p_value
        topic[PEARSON_CORRELATION] = pearson_correlation
        topic[PEARSON_P_VALUE] = pearson_p_value
        topic[PREVIOUS_AVERAGE_AUDIENCE] = previous_average
        topic[CURRENT_AVERAGE_AUDIENCE] = current_average

    df = df.append(rows)
    df = df.sort_values(by=[INTEREST, GENDER], ascending=False)

    workbook = writer.book
    format1 = workbook.add_format({'num_format': '#,##0.0000000000'})
    df[[INTEREST, COUNTRY_CODE, GENDER, SPEARMAN_CORRELATION, SPEARMAN_P_VALUE, PEARSON_CORRELATION, PEARSON_P_VALUE, PREVIOUS_RANK_RELATIVE, CURRENT_RANK_RELATIVE, PREVIOUS_AVERAGE_AUDIENCE, CURRENT_AVERAGE_AUDIENCE, PREVIOUS_RANK, CURRENT_RANK]].to_excel(writer, sheet_name=name)
    print_summary_results(df,writer,"Summary-" + name)
    writer.sheets[name].set_column('A:F', 18, format1)
    writer.sheets[name].set_column('G:J', 30, format1)
    return df


def get_range_position_given_list_audiences_data_and_age_range(age_range, audience_data_list):
    for index in range(0, len(audience_data_list)):
        aux_audience_data = audience_data_list[index]
        if aux_audience_data[MIN_AGE] == age_range[0] and aux_audience_data[MAX_AGE] == age_range[1]:
            return index
    raise Exception("Should find the index:" + str(age_range))

def get_value_from_audiences_data_and_age_range(age_range, audience_data_list):
    for index in range(0, len(audience_data_list)):
        aux_audience_data = audience_data_list[index]
        if aux_audience_data[MIN_AGE] == age_range[0] and aux_audience_data[MAX_AGE] == age_range[1]:
            return aux_audience_data[VALUE]
    raise Exception("Should find the index:" + str(age_range))

def get_value_from_audiences_data_and_gender(gender, audience_data_list):
    for index in range(0, len(audience_data_list)):
        aux_audience_data = audience_data_list[index]
        if aux_audience_data[GENDER] == gender:
            return aux_audience_data[VALUE]
    raise Exception("Should find the index:" + str(gender))

def get_gender_rank_given_list_audience_data(audience_data_list):
    rank = []
    audience_data_list = sorted(audience_data_list, key=lambda audience_data: audience_data[VALUE])
    genders = [1,2]
    # if len(audience_data_list) != len(ranges):
    #     import ipdb;ipdb.set_trace()
    for gender in genders:
        rank.append(get_value_from_audiences_data_and_gender(gender,audience_data_list))
    return rank

def get_age_rank_given_list_audience_data(audience_data_list):
    rank = []
    audience_data_list = sorted(audience_data_list, key=lambda audience_data: audience_data[VALUE])
    ranges = [(18, 24),
              (25, 29),
              (30, 34),
              (35, 39),
              (40, 44),
              (45, 49),
              (50, 54),
              (55, 59)]
    # if len(audience_data_list) != len(ranges):
    #     import ipdb;ipdb.set_trace()
    for range in ranges:
        rank.append(get_value_from_audiences_data_and_age_range(range,audience_data_list))
    return rank

def get_interest_from_row(row):
    if pd.isnull(row["interests"]):
        return None
    else:
        name = ast.literal_eval(row["interests"])["name"]
        return name

# Create a Pandas Excel writer using XlsxWriter as the engine.
writer = pd.ExcelWriter('/tmp/rank_correlations.xlsx', engine='xlsxwriter', options={'nan_inf_to_errors': False})


# Loading data
new_data = pd.DataFrame.from_csv("/tmp/dataframe_recollection_health_awareness_main.csv")
new_data = new_data[new_data["ages_ranges"] != "{u'min': 18}"]
new_data = new_data[new_data["ages_ranges"] != "{u'max': 24, u'min': 20}"]
new_data = new_data[new_data["genders"] == 0]

new_data_18p = pd.DataFrame.from_csv("/tmp/dataframe_recollection_health_awareness_main.csv")
new_data_18p = new_data_18p[new_data_18p["ages_ranges"] == "{u'min': 18}"]
new_data_18p = new_data_18p[new_data_18p["genders"] == 0]

new_data_18p[INTEREST] = new_data_18p.apply(lambda row: get_interest_from_row(row), axis=1)

old_data = pd.DataFrame.from_csv("/home/maraujo/Dropbox/qcri/prototype/pyads_audience/final_www_data/current_experiment.csv.gz")
country_data = pd.DataFrame.from_csv("/home/maraujo/Dropbox/qcri/prototype/pyads_audience/final_www_data/country_data.csv")

old_data = old_data[pd.isnull(old_data["placebo_query"])] #Remove placebo queries from the past
old_data = old_data[old_data["country_code"].isin(['US','IN','BR','ID','MX','PH','TR','GB','TH','VN'])] #Filter just top 10 countries
old_data = old_data[(
                        ((old_data[MIN_AGE] == 18) & (old_data[MAX_AGE] == 24)) |
                        ((old_data[MIN_AGE] == 25) & (old_data[MAX_AGE] == 29)) |
                        ((old_data[MIN_AGE] == 30) & (old_data[MAX_AGE] == 34)) |
                        ((old_data[MIN_AGE] == 35) & (old_data[MAX_AGE] == 39)) |
                        ((old_data[MIN_AGE] == 40) & (old_data[MAX_AGE] == 44)) |
                        ((old_data[MIN_AGE] == 45) & (old_data[MAX_AGE] == 49)) |
                        ((old_data[MIN_AGE] == 50) & (old_data[MAX_AGE] == 54)) |
                        ((old_data[MIN_AGE] == 55) & (old_data[MAX_AGE] == 59)))
                    ]
old_data = old_data[old_data[GENDER] == 0]

#Loop through old_data and create countries with old audience
old_data.apply(lambda row: add_old_row(row), axis=1)

# Populate with new data country audience
new_data.apply(lambda row: add_new_row(row), axis=1)

#Output
output_dataframe = pd.DataFrame(columns=[COUNTRY_CODE, INTEREST, PREVIOUS_AUDIENCE, CURRENT_AUDIENCE, ABSOLUTE_VARIATION, RELATIVE_VARIATION, NORMALIZED_VARIATION, SIGNED_VARIATION, SIGNED_VARIATION_PERCENTAGE])
# Calculate Relative Audiences
for country in countries.values():
    country.calculate_relative_audiences()
# Calculate variations
for country in countries.values():
    country.process_comparison()
    output_dataframe = output_dataframe.append(country.get_data_frame_rows())
output_dataframe.sort_values(by=[COUNTRY_CODE, INTEREST, MIN_AGE, MAX_AGE, GENDER])
output_dataframe[[COUNTRY_CODE, INTEREST, MIN_AGE, MAX_AGE, GENDER, PREVIOUS_AUDIENCE, CURRENT_AUDIENCE, ABSOLUTE_VARIATION, RELATIVE_VARIATION, NORMALIZED_VARIATION, SIGNED_VARIATION, SIGNED_VARIATION_PERCENTAGE]].to_excel("/tmp/recollectComparison.xlsx")

#Get ranks by age range
TOPICS_RANK_BY_AGE_RANGE = []
for country in countries.values():
    for gender in [0]:
            new_audiences_data = filter(lambda audience_data: audience_data[GENDER] == gender, country.audiences[CURRENT_AUDIENCE])
            old_audiences_data = filter(lambda audience_data: audience_data[GENDER] == gender, country.audiences[PREVIOUS_AUDIENCE])
            new_relative_audiences_data = filter(lambda audience_data: audience_data[GENDER] == gender, country.audiences[CURRENT_AUDIENCE])
            old_relative_audiences_data = filter(lambda audience_data: audience_data[GENDER] == gender, country.audiences[PREVIOUS_AUDIENCE])
            TOPICS_RANK_BY_AGE_RANGE.append({COUNTRY_CODE: country.code,
                                             INTEREST: None,
                                             GENDER: gender,
                                "ids": None,
                                             PREVIOUS_RANK : get_age_rank_given_list_audience_data(old_audiences_data),
                                             CURRENT_RANK: get_age_rank_given_list_audience_data(new_audiences_data),
                                             PREVIOUS_RANK_RELATIVE: get_age_rank_given_list_audience_data(old_relative_audiences_data),
                                             CURRENT_RANK_RELATIVE: get_age_rank_given_list_audience_data(new_relative_audiences_data)})
    for topic in country.topics_audiences:
        for gender in [0]:
            new_audiences_topic_data = filter(lambda audience_data: audience_data[GENDER] == gender, topic[CURRENT_AUDIENCE])
            old_audiences_topic_data = filter(lambda audience_data: audience_data[GENDER] == gender, topic[PREVIOUS_AUDIENCE])
            new_relative_audiences_topic_data = filter(lambda audience_data: audience_data[GENDER] == gender, topic[CURRENT_RELATIVE_AUDIENCE])
            old_relative_audiences_topic_data = filter(lambda audience_data: audience_data[GENDER] == gender, topic[PREVIOUS_RELATIVE_AUDIENCE])
            TOPICS_RANK_BY_AGE_RANGE.append({COUNTRY_CODE: country.code,
                                             INTEREST: topic[INTEREST],
                                             GENDER: gender,
                                "ids": topic["ids"],
                                             PREVIOUS_RANK : get_age_rank_given_list_audience_data(old_audiences_topic_data),
                                             CURRENT_RANK: get_age_rank_given_list_audience_data(new_audiences_topic_data),
                                             PREVIOUS_RANK_RELATIVE: get_age_rank_given_list_audience_data(old_relative_audiences_topic_data),
                                             CURRENT_RANK_RELATIVE: get_age_rank_given_list_audience_data(new_relative_audiences_topic_data)})

# Create a Pandas Excel writer using XlsxWriter as the engine.
writer = pd.ExcelWriter('/tmp/rank_correlations.xlsx', engine='xlsxwriter')

# Calculate spearman's rank correlation among all countries
list_of_countries = CountryData.get_all_countries_code()
build_correlation_dataframe_given_topics(TOPICS_RANK_BY_AGE_RANGE, "by Age Range", writer, min_initial_relative_audience=0)
compare_two_contries_movement(CountryData.get_country("US"),CountryData.get_country("GB"),writer)
compare_two_contries_movement(CountryData.get_country("US"),CountryData.get_country("IN"),writer)
compare_two_contries_movement(CountryData.get_country("US"),CountryData.get_country("BR"),writer)

# ==================================================================================================================================================================
# By gender
new_data = pd.DataFrame.from_csv("/tmp/dataframe_recollection_health_awareness_main.csv")
new_data = new_data[new_data["ages_ranges"] == "{u'min': 18}"]
countries = {}
old_data = pd.DataFrame.from_csv("/home/maraujo/Dropbox/qcri/prototype/pyads_audience/final_www_data/current_experiment.csv.gz")
old_data = old_data[pd.isnull(old_data["placebo_query"])] #Remove placebo queries from the past
old_data = old_data[old_data["country_code"].isin(['US','IN','BR','ID','MX','PH','TR','GB','TH','VN'])] #Filter just top 10 countries
old_data = old_data[((old_data[MIN_AGE] == 18) & (pd.isnull(old_data[MAX_AGE])))]
#Loop through old_data and create countries with old audience
old_data.apply(lambda row: add_old_row(row), axis=1)

# Populate with new data country audience
new_data.apply(lambda row: add_new_row(row), axis=1)
#Output
output_dataframe = pd.DataFrame(columns=[COUNTRY_CODE, INTEREST, PREVIOUS_AUDIENCE, CURRENT_AUDIENCE, ABSOLUTE_VARIATION, RELATIVE_VARIATION, NORMALIZED_VARIATION])
# Calculate Relative Audiences
for country in countries.values():
    country.calculate_relative_audiences()
# Calculate variations
for country in countries.values():
    country.process_comparison()
    output_dataframe = output_dataframe.append(country.get_data_frame_rows())
output_dataframe[[COUNTRY_CODE, INTEREST, PREVIOUS_AUDIENCE, CURRENT_AUDIENCE, ABSOLUTE_VARIATION, RELATIVE_VARIATION, NORMALIZED_VARIATION]].to_excel("/tmp/recollectComparison_Gender.xlsx")

# #Get ranks by age range
TOPICS_RANK_BY_GENDER = []
for country in countries.values():
    for topic in country.topics_audiences:
            new_audiences_topic_data = topic[CURRENT_AUDIENCE]
            old_audiences_topic_data = topic[PREVIOUS_AUDIENCE]
            new_relative_audiences_topic_data = topic[CURRENT_RELATIVE_AUDIENCE]
            old_relative_audiences_topic_data = topic[PREVIOUS_RELATIVE_AUDIENCE]
            TOPICS_RANK_BY_GENDER.append({COUNTRY_CODE: country.code,
                                             INTEREST: topic[INTEREST],
                                "ids": topic["ids"],
                                             PREVIOUS_RANK : get_gender_rank_given_list_audience_data(old_audiences_topic_data),
                                             CURRENT_RANK: get_gender_rank_given_list_audience_data(new_audiences_topic_data),
                                             PREVIOUS_RANK_RELATIVE: get_gender_rank_given_list_audience_data(old_relative_audiences_topic_data),
                                             CURRENT_RANK_RELATIVE: get_gender_rank_given_list_audience_data(new_relative_audiences_topic_data)})

# Calculate spearman's rank correlation among all countries
list_of_countries = CountryData.get_all_countries_code()
build_correlation_dataframe_given_topics(TOPICS_RANK_BY_GENDER, "by Gender", writer, min_initial_relative_audience=0)

# #English Speaking Countries
# list_of_countries = country_data[country_data["highfb_english"]]["country_code"].values.tolist()
# build_correlation_dataframe_given_countries_topics(TOPICS_RANK, list_of_countries, "English Speakers", writer, min_initial_relative_audience=0)
# #OECD
# list_of_countries = country_data[country_data["oecd"]]["country_code"].values.tolist()
# build_correlation_dataframe_given_countries_topics(TOPICS_RANK, list_of_countries, "OECD", writer, min_initial_relative_audience=0)
#
# #FB Audience > 10M
# list_of_countries = country_data[country_data["facebook_population"] > 10000000]["country_code"].values.tolist()
# build_correlation_dataframe_given_countries_topics(TOPICS_RANK, list_of_countries, "FB Audience > 10M", writer, min_initial_relative_audience=0)
#
# #North America or Europe
# list_of_countries = country_data[country_data["n_america"] | country_data["europe"]]["country_code"].values.tolist()
# build_correlation_dataframe_given_countries_topics(TOPICS_RANK, list_of_countries, "N. America or Europe", writer, min_initial_relative_audience=0)
#
# # -------------------- 0.01 min initial relative audience
#
# #North America or Europe
# list_of_countries = country_data[country_data["developed"]]["country_code"].values.tolist()
# build_correlation_dataframe_given_countries_topics(TOPICS_RANK, list_of_countries, "Developed", writer, min_initial_relative_audience=0.01)
#
# #TOP 20 Gdp
# list_of_countries = country_data.sort_values(by="gdp", ascending=False)[:20]["country_code"].values.tolist()
# build_correlation_dataframe_given_countries_topics(TOPICS_RANK, list_of_countries, "TOP 20 gdp", writer, min_initial_relative_audience=0.01)
#
# # Calculate spearman's rank correlation among all countries
# list_of_countries = CountryData.get_all_countries_code()
# build_correlation_dataframe_given_countries_topics(TOPICS_RANK, list_of_countries, "All Countries", writer, min_initial_relative_audience=0.01)
#
# #English Speaking Countries
# list_of_countries = country_data[country_data["highfb_english"]]["country_code"].values.tolist()
# build_correlation_dataframe_given_countries_topics(TOPICS_RANK, list_of_countries, "English Speakers", writer, min_initial_relative_audience=0.01)
# #OECD
# list_of_countries = country_data[country_data["oecd"]]["country_code"].values.tolist()
# build_correlation_dataframe_given_countries_topics(TOPICS_RANK, list_of_countries, "OECD", writer, min_initial_relative_audience=0.01)
#
# #FB Audience > 10M
# list_of_countries = country_data[country_data["facebook_population"] > 10000000]["country_code"].values.tolist()
# build_correlation_dataframe_given_countries_topics(TOPICS_RANK, list_of_countries, "FB Audience > 10M", writer, min_initial_relative_audience=0.01)
#
# #North America or Europe
# list_of_countries = country_data[country_data["n_america"] | country_data["europe"]]["country_code"].values.tolist()
# build_correlation_dataframe_given_countries_topics(TOPICS_RANK, list_of_countries, "N. America or Europe", writer, min_initial_relative_audience=0.01)
#
# #North America or Europe
# list_of_countries = country_data[country_data["developed"]]["country_code"].values.tolist()
# build_correlation_dataframe_given_countries_topics(TOPICS_RANK, list_of_countries, "Developed", writer, min_initial_relative_audience=0.01)
#
# #TOP 20 Gdp
# list_of_countries = country_data.sort_values(by="gdp", ascending=False)[:20]["country_code"].values.tolist()
# build_correlation_dataframe_given_countries_topics(TOPICS_RANK, list_of_countries, "TOP 20 gdp", writer, min_initial_relative_audience=0.01)
#
# # -------------------- 0.05 min initial relative audience
#
# #North America or Europe
# list_of_countries = country_data[country_data["developed"]]["country_code"].values.tolist()
# build_correlation_dataframe_given_countries_topics(TOPICS_RANK, list_of_countries, "Developed", writer, min_initial_relative_audience=0.05)
#
# #TOP 20 Gdp
# list_of_countries = country_data.sort_values(by="gdp", ascending=False)[:20]["country_code"].values.tolist()
# build_correlation_dataframe_given_countries_topics(TOPICS_RANK, list_of_countries, "TOP 20 gdp", writer, min_initial_relative_audience=0.05)
#
# # Calculate spearman's rank correlation among all countries
# list_of_countries = CountryData.get_all_countries_code()
# build_correlation_dataframe_given_countries_topics(TOPICS_RANK, list_of_countries, "All Countries", writer, min_initial_relative_audience=0.05)
#
# #English Speaking Countries
# list_of_countries = country_data[country_data["highfb_english"]]["country_code"].values.tolist()
# build_correlation_dataframe_given_countries_topics(TOPICS_RANK, list_of_countries, "English Speakers", writer, min_initial_relative_audience=0.05)
# #OECD
# list_of_countries = country_data[country_data["oecd"]]["country_code"].values.tolist()
# build_correlation_dataframe_given_countries_topics(TOPICS_RANK, list_of_countries, "OECD", writer, min_initial_relative_audience=0.05)
#
# #FB Audience > 10M
# list_of_countries = country_data[country_data["facebook_population"] > 10000000]["country_code"].values.tolist()
# build_correlation_dataframe_given_countries_topics(TOPICS_RANK, list_of_countries, "FB Audience > 10M", writer, min_initial_relative_audience=0.05)
#
# #North America or Europe
# list_of_countries = country_data[country_data["n_america"] | country_data["europe"]]["country_code"].values.tolist()
# build_correlation_dataframe_given_countries_topics(TOPICS_RANK, list_of_countries, "N. America or Europe", writer, min_initial_relative_audience=0.05)
#
# #North America or Europe
# list_of_countries = country_data[country_data["developed"]]["country_code"].values.tolist()
# build_correlation_dataframe_given_countries_topics(TOPICS_RANK, list_of_countries, "Developed", writer, min_initial_relative_audience=0.05)
#
# #TOP 20 Gdp
# list_of_countries = country_data.sort_values(by="gdp", ascending=False)[:20]["country_code"].values.tolist()
# build_correlation_dataframe_given_countries_topics(TOPICS_RANK, list_of_countries, "TOP 20 gdp", writer, min_initial_relative_audience=0.05)



#save excel file
writer.save()