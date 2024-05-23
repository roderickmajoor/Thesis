import xml.etree.ElementTree as ET
from shapely.geometry import Polygon

def parse_page_xml(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    ns = {'page': 'http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15'}

    table_cells_polygons = []

    for table_cell in root.findall('.//page:TableCell', ns):
        coords = table_cell.find('page:Coords', ns).attrib['points']
        points = [tuple(map(int, point.split(','))) for point in coords.split()]
        polygon = Polygon(points)
        table_cells_polygons.append(polygon)

    return table_cells_polygons

# Example usage:
#xml_file = '/home/roderickmajoor/Desktop/Master/Thesis/GT_data/55/page/WBMA00007000010.xml'
#table_cells_polygons = parse_page_xml(xml_file)

#print(len(table_cells_polygons))